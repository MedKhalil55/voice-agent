"""Synchronous MCP client for the ACM Banking Voice Agent.

This client launches `mcp_server/server.py` as a subprocess (stdio transport)
and exposes a SYNCHRONOUS `call_tool()` method that is safe to call from
LangGraph synchronous nodes.

A persistent background thread runs a dedicated asyncio event loop which owns
the MCP session. Synchronous calls are marshalled onto that loop via
`asyncio.run_coroutine_threadsafe`, keeping a single long-lived subprocess
alive for the lifetime of the process.
"""

import asyncio
import json
import sys
import threading
from time import strftime
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _log(message: str) -> None:
    """Timestamped, flushed logger so MCP activity shows up in FastAPI logs."""
    print(f"[{strftime('%H:%M:%S')}] {message}", flush=True)



class ACMMCPClient:
    """Singleton synchronous wrapper around the MCP banking server."""

    _instance: Optional["ACMMCPClient"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._ready = threading.Event()
        self._connected = False

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "ACMMCPClient":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Background event loop / connection management
    # ------------------------------------------------------------------
    def _ensure_connected(self) -> None:
        """Lazily start the background loop and connect to the MCP server."""
        if self._connected:
            return

        with self._lock:
            if self._connected:
                return

            self._ready.clear()
            self._thread = threading.Thread(
                target=self._loop_runner, name="acm-mcp-client", daemon=True
            )
            self._thread.start()

            # Wait for the background loop + session to be ready.
            if not self._ready.wait(timeout=60):
                raise RuntimeError("Timed out connecting to MCP server")

            if not self._connected:
                raise RuntimeError("Failed to connect to MCP server")

    def _loop_runner(self) -> None:
        """Runs in the background thread: owns the event loop + MCP session."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except Exception as exc:  # noqa: BLE001
            _log(f"[ACMMCPClient] background loop error: {exc}")

        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _serve(self) -> None:
        """Open the stdio connection and keep the session alive forever."""
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "mcp_server/server.py"],
        )

        self._stop_event = asyncio.Event()
        try:
            # Forward the MCP server subprocess's stderr to this process's
            # stderr so its [MCP-SERVER] tool-execution logs appear live in the
            # FastAPI terminal.
            async with stdio_client(server_params, errlog=sys.stderr) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._session = session
                    self._connected = True
                    _log("[ACMMCPClient] connected to MCP server")
                    self._ready.set()

                    # Keep the session open until shutdown is requested.
                    await self._stop_event.wait()
        except Exception as exc:  # noqa: BLE001
            _log(f"[ACMMCPClient] connection error: {exc}")
            self._connected = False
            self._ready.set()


    # ------------------------------------------------------------------
    # Public synchronous API
    # ------------------------------------------------------------------
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Synchronous wrapper — safe to call from LangGraph nodes."""
        arguments = arguments or {}
        _log(f"[MCP] → call_tool({tool_name}, {arguments})")
        try:
            self._ensure_connected()
            future = asyncio.run_coroutine_threadsafe(
                self._call_tool_async(tool_name, arguments),
                self._loop,
            )
            return future.result(timeout=30)
        except Exception as exc:  # noqa: BLE001
            _log(f"[MCP] ✗ call_tool error ({tool_name}): {exc}")
            return {"error": str(exc), "ok": False}


    async def _call_tool_async(self, tool_name: str, arguments: dict) -> dict:
        if self._session is None:
            raise RuntimeError("MCP session not initialized")

        response = await self._session.call_tool(tool_name, arguments)

        # Extract the first text content payload and parse JSON.
        for content in response.content:
            text = getattr(content, "text", None)
            if text is not None:
                try:
                    parsed = json.loads(text)
                    _log(f"[MCP] ✓ tool_result({tool_name}) = {parsed}")
                    if isinstance(parsed, dict):
                        return parsed
                    return {"result": parsed}
                except json.JSONDecodeError:
                    _log(f"[MCP] ✓ tool_result({tool_name}) raw_text = {text}")
                    return {"result": text}

        _log(f"[MCP] ✗ tool_result({tool_name}) empty_response")

        return {"error": "empty_response", "ok": False}

    # ------------------------------------------------------------------
    # Optional shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if self._loop is not None and self._connected:
            try:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            except Exception:
                pass
        self._connected = False
