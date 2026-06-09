import asyncio
import json
import uuid
from datetime import date, datetime
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import RealDictCursor
from starlette.concurrency import run_in_threadpool

from api.voice_agent_ws import WebSocketVoiceAgent

from db.database import get_connection
from db.tools import get_client_info
from llm.langgraph_agent import classify_client_profile

load_dotenv(override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Active calls registry
# key: call_id (str uuid)
# value: dict with agent_ws, client_ws, customer_id, status
active_calls: dict[str, dict] = {}
active_calls_lock = asyncio.Lock()

# Active voice agent sessions
# key: call_id, value: WebSocketVoiceAgent instance
agent_sessions: dict[str, WebSocketVoiceAgent] = {}


def _safe_set_status(call_id: str, status: str) -> None:
    call = active_calls.get(call_id)
    if call is None:
        return
    call["status"] = status


async def _free_call_status_db(customer_id: int, reason: str) -> None:
    """Release call status in DB after reject or end."""
    try:
        from db.tools import set_call_status

        await run_in_threadpool(
            set_call_status,
            customer_id,
            "FREE",
            None,
            "",
            reason,
        )
    except Exception:
        pass


async def _update_status_from_message_async(call_id: str, message_text: str) -> None:
    try:
        payload = json.loads(message_text)
    except Exception:
        return

    if not isinstance(payload, dict):
        return

    msg_type = payload.get("type")

    if msg_type == "call_accepted":
        _safe_set_status(call_id, "accepted")

    elif msg_type == "call_rejected":
        _safe_set_status(call_id, "rejected")
        customer_id = active_calls.get(call_id, {}).get("customer_id")
        if customer_id:
            await _free_call_status_db(customer_id, "Appel rejeté depuis émulateur")

    elif msg_type == "call_ended":
        _safe_set_status(call_id, "ended")
        customer_id = active_calls.get(call_id, {}).get("customer_id")
        if customer_id and call_id not in agent_sessions:
            await _free_call_status_db(customer_id, "Appel terminé côté client")

    elif msg_type == "ringing":
        _safe_set_status(call_id, "ringing")


async def relay_messages(sender: WebSocket, receiver_getter, call_id: str, side: str):
    try:
        while True:
            data = await sender.receive_text()
            await _update_status_from_message_async(call_id, data)
            receiver = receiver_getter()
            if receiver:
                await receiver.send_text(data)

    except WebSocketDisconnect:
        # ← MODIFIER : ne pas écraser "in_call" quand le dashboard se déconnecte
        # Un refresh de page = déconnexion temporaire, pas fin d'appel
        current = active_calls.get(call_id, {}).get("status", "")
        if current not in ("rejected", "in_call", "accepted"):
            _safe_set_status(call_id, "ended")


@app.websocket("/ws/agent/{call_id}")
async def ws_agent(websocket: WebSocket, call_id: str):
    await websocket.accept()

    # Créer ou récupérer l'entrée existante
    call = active_calls.setdefault(
        call_id,
        {
            "customer_id": None,
            "status": "ringing",
            "agent_ws": None,
            "client_ws": None,
            "transcripts": [],
        },
    )

    # ← Remplacer l'ancien agent_ws par le nouveau (reconnexion)
    old_ws = call.get("agent_ws")
    if old_ws is not None and old_ws is not websocket:
        try:
            await old_ws.close()
        except Exception:
            pass

    call["agent_ws"] = websocket

    # ← Si appel en cours, envoyer l'état actuel au dashboard reconnecté
    current_status = call.get("status", "ringing")
    try:
        await websocket.send_json(
            {
                "type": "call_status",
                "status": current_status,
                "call_id": call_id,
            }
        )
    except Exception:
        pass

    # Replay past transcripts to the dashboard after reconnect.
    past_transcripts = call.get("transcripts", [])
    for transcript_msg in past_transcripts:
        try:
            await websocket.send_json(transcript_msg)
        except Exception:
            break

    # Envoyer ringing au client si connecté
    client_ws = call.get("client_ws")
    if client_ws is not None and current_status == "ringing":
        try:
            await client_ws.send_text(json.dumps({"type": "ringing"}))
        except Exception:
            pass

    def _get_client_ws():
        return active_calls.get(call_id, {}).get("client_ws")

    try:
        await relay_messages(websocket, _get_client_ws, call_id, side="agent")
    finally:
        existing = active_calls.get(call_id)
        if existing is not None and existing.get("agent_ws") is websocket:
            existing["agent_ws"] = None


@app.websocket("/ws/client/{call_id}")
async def ws_client(websocket: WebSocket, call_id: str):
    await websocket.accept()

    call = active_calls.setdefault(
        call_id,
        {
            "customer_id": None,
            "status": "ringing",
            "agent_ws": None,
            "client_ws": None,
            "transcripts": [],
        },
    )
    call["client_ws"] = websocket

    def _get_agent_ws():
        return active_calls.get(call_id, {}).get("agent_ws")

    try:
        await relay_messages(websocket, _get_agent_ws, call_id, side="client")
    finally:
        existing = active_calls.get(call_id)
        if existing is not None and existing.get("client_ws") is websocket:
            existing["client_ws"] = None


@app.websocket("/ws/audio/{call_id}")
async def audio_stream(websocket: WebSocket, call_id: str):
    """Audio WebSocket for VoiceAgent integration.

    Protocol:
    - Client sends: binary audio chunks (PCM 16kHz 16bit mono)
    - Client sends: JSON {"type": "start"} to initiate agent
    - Client sends: JSON {"type": "stop"} to end call
    - Server sends: binary audio chunks (WAV TTS response)
    - Server sends: JSON {"type": "transcript", "role": "client"|"agent", "text": "..."}
    """

    await websocket.accept()

    call_info = active_calls.get(call_id)
    if not call_info:
        await websocket.send_json({"type": "error", "message": "Call not found"})
        await websocket.close()
        return

    customer_id = call_info.get("customer_id")
    if customer_id is None:
        await websocket.send_json(
            {"type": "error", "message": "Missing customer_id for call"}
        )
        await websocket.close()
        return
    agent: WebSocketVoiceAgent | None = None

    # Event set when the agent requests shutdown (hang up).
    agent_shutdown_event = asyncio.Event()

    async def send_audio(audio_bytes: bytes) -> None:
        """Send TTS audio back to client browser."""

        try:
            await websocket.send_bytes(audio_bytes)
        except Exception:
            pass

    async def send_transcript(event: dict) -> None:
        """Send transcript events to client (and optionally to agent dashboard)."""

        try:
            payload = {
                "type": "transcript",
                "role": "agent" if event.get("type") == "agent_speech" else "client",
                "text": event.get("text", ""),
            }
            call_data = active_calls.get(call_id)
            if call_data is not None:
                call_data.setdefault("transcripts", []).append(payload)
            await websocket.send_json(payload)

            # Also relay to agent dashboard WebSocket if connected.
            agent_ws = active_calls.get(call_id, {}).get("agent_ws")
            if agent_ws:
                try:
                    await agent_ws.send_json(payload)
                except Exception:
                    pass
        except Exception:
            pass

    async def on_agent_shutdown() -> None:
        """Called by VoiceAgent when it wants to hang up."""

        from main import _log

        _log("[WS-AUDIO] Agent requested shutdown — closing WebSocket")

        # Release DB call status — the agent hung up.
        await _free_call_status_db(customer_id, "Agent a raccroché")

        # Notify Angular client
        try:
            await websocket.send_json({"type": "call_ended"})

        except Exception:
            pass

        # Notify agent dashboard
        agent_ws = active_calls.get(call_id, {}).get("agent_ws")
        if agent_ws:
            try:
                await agent_ws.send_json({"type": "call_ended"})
            except Exception:
                pass

        # Notify signaling channel (phone emulator)
        client_ws = active_calls.get(call_id, {}).get("client_ws")
        if client_ws:
            try:
                await client_ws.send_text(json.dumps({"type": "call_ended"}))
            except Exception:
                pass

        active_calls[call_id]["status"] = "ended"
        agent_shutdown_event.set()

    try:
        while True:
            receive_task = asyncio.create_task(websocket.receive())
            shutdown_task = asyncio.create_task(agent_shutdown_event.wait())

            done, pending = await asyncio.wait(
                [receive_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            if shutdown_task in done:
                break

            if receive_task not in done:
                continue

            try:
                message = receive_task.result()
            except Exception:
                break

            if message.get("type") == "websocket.disconnect":
                break

            # JSON control message
            if message.get("text") is not None:
                data = json.loads(message["text"])

                if data.get("type") == "start":
                    # Initialize VoiceAgent for this call
                    if call_id not in agent_sessions:
                        agent = WebSocketVoiceAgent(
                            customer_id=int(customer_id),
                            send_audio_callback=send_audio,
                            send_transcript_callback=send_transcript,
                            on_shutdown_callback=on_agent_shutdown,
                        )
                        agent_sessions[call_id] = agent

                        # Start agent in background thread
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, agent.start_ws)

                        active_calls[call_id]["status"] = "in_call"
                    else:
                        agent = agent_sessions[call_id]

                elif data.get("type") == "stop":
                    # End call - cleanup
                    if call_id in agent_sessions:
                        agent_sessions[call_id].shutdown()
                        del agent_sessions[call_id]
                    active_calls[call_id]["status"] = "ended"
                    await _free_call_status_db(
                        customer_id, "Appel arrêté par le client"
                    )
                    break


            # Binary audio chunk from client microphone
            elif message.get("bytes") is not None:
                pcm_data = message["bytes"]
                if agent and pcm_data:
                    await agent.process_audio_chunk(pcm_data)

    except WebSocketDisconnect:
        pass
    finally:
        current_status = active_calls.get(call_id, {}).get("status", "ended")
        is_real_end = current_status in ("ended", "rejected")

        if is_real_end:
            if call_id in agent_sessions:
                try:
                    agent_sessions[call_id].shutdown()
                except Exception:
                    pass
                del agent_sessions[call_id]

        try:
            await websocket.close()
        except Exception:
            pass


def _fetch_all(sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
    """Run a SELECT query using the existing pooled connection helper."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


# -------------------------
# Pydantic models
# -------------------------

try:
    # Pydantic v2
    from pydantic import BaseModel, ConfigDict

    class _BaseModel(BaseModel):
        model_config = ConfigDict(from_attributes=True)

except Exception:  # pragma: no cover
    # Pydantic v1 fallback
    from pydantic import BaseModel

    class _BaseModel(BaseModel):
        class Config:
            orm_mode = True


class ClientSummary(_BaseModel):
    customer_id: int
    customer_name: Optional[str] = None
    unpaid_amount: Optional[float] = None
    late_days: Optional[int] = None
    statut_workflow: Optional[str] = None
    telephone_1: Optional[str] = None
    email: Optional[str] = None
    profile_type: Optional[str] = None
    call_status: Optional[str] = (
        None  # FREE/PROMISED/REFUSED/CALLBACK/IN_CALL/BROKEN/KEPT
    )
    next_call_date: Optional[str] = None  # ISO date string
    days_left: Optional[int] = None
    can_call: Optional[bool] = True


class CallLogEntry(_BaseModel):
    id: int
    customer_id: int
    transcript: Optional[str] = None
    intent: Optional[str] = None
    outcome: Optional[str] = None
    agent_decision: Optional[str] = None
    session_id: Optional[str] = None
    turn_number: Optional[int] = None
    call_date: Optional[datetime] = None


class PaymentPromiseEntry(_BaseModel):
    id: int
    customer_id: int
    amount: Optional[float] = None
    installments: Optional[int] = None
    promised_date: Optional[date] = None
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    reason: Optional[str] = None
    reason_raw: Optional[str] = None


class ClaimEntry(_BaseModel):
    id_acm_claims: int
    customer_id: int
    subject: Optional[str] = None
    body: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    status: Optional[str] = None
    date_insertion: Optional[date] = None


class CallInitiateRequest(_BaseModel):
    customer_id: int


class CallInitiateResponse(_BaseModel):
    session_id: str
    call_id: str
    customer_id: int
    status: str
    customer_name: Optional[str] = None


class CallStatusResponse(_BaseModel):
    call_id: str
    status: str
    customer_id: Optional[int] = None


# -------------------------
# Endpoints
# -------------------------


@app.get("/api/clients", response_model=list[ClientSummary])
async def api_get_clients() -> list[ClientSummary]:
    try:
        rows = await run_in_threadpool(
            _fetch_all,
            """
            SELECT DISTINCT c.customer_id_extern as customer_id
            FROM acm_customer c
            JOIN acm_collection col ON c.customer_id_extern = col.customer_id_extern
            WHERE col.acm_enabled = TRUE
            ORDER BY c.customer_id_extern
            """,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}")

    results: list[ClientSummary] = []
    for row in rows:
        customer_id = row["customer_id"]
        try:
            info = await run_in_threadpool(get_client_info, customer_id)
            if not info.get("found"):
                continue
            profile = await run_in_threadpool(classify_client_profile, info)
            from db.tools import get_call_status

            cs = await run_in_threadpool(get_call_status, customer_id)
            results.append(
                ClientSummary(
                    customer_id=customer_id,
                    customer_name=info.get("customer_name"),
                    unpaid_amount=float(info.get("unpaid_amount") or 0),
                    late_days=int(info.get("late_days") or 0),
                    statut_workflow=info.get("statut_workflow"),
                    telephone_1=info.get("telephone_1"),
                    email=info.get("email"),
                    profile_type=profile.get("profile"),
                    call_status=cs.get("status", "FREE"),
                    next_call_date=cs.get("next_call_date"),
                    days_left=cs.get("days_left"),
                    can_call=cs.get("can_call", True),
                )
            )
        except Exception:
            continue
    return results


@app.get("/api/call-status/{customer_id}")
async def api_get_customer_call_status(customer_id: int):
    from db.tools import get_call_status

    result = await run_in_threadpool(get_call_status, customer_id)
    return result


@app.get("/api/calls", response_model=list[CallLogEntry])
async def api_get_calls(
    customer_id: int | None = Query(default=None),
) -> list[CallLogEntry]:
    sql = """
    SELECT
        id,
        customer_id,
        transcript,
        intent,
        outcome,
        agent_decision,
        session_id,
        turn_number,
        call_date
    FROM acm_call_log
    """

    params: tuple[Any, ...] | None = None
    if customer_id is not None:
        sql += " WHERE customer_id = %s"
        params = (customer_id,)

    sql += " ORDER BY call_date DESC LIMIT 100"

    try:
        rows = await run_in_threadpool(_fetch_all, sql, params)
        return [CallLogEntry(**row) for row in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc


@app.get("/api/promises", response_model=list[PaymentPromiseEntry])
async def api_get_promises(
    customer_id: int | None = Query(default=None),
) -> list[PaymentPromiseEntry]:
    sql = """
    SELECT
        id,
        customer_id,
        amount,
        installments,
        promised_date,
        status,
        created_at,
        reason,
        reason_raw
    FROM acm_payment_promise
    """

    params: tuple[Any, ...] | None = None
    if customer_id is not None:
        sql += " WHERE customer_id = %s"
        params = (customer_id,)

    try:
        rows = await run_in_threadpool(_fetch_all, sql, params)
        return [PaymentPromiseEntry(**row) for row in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc


@app.get("/api/claims", response_model=list[ClaimEntry])
async def api_get_claims(
    customer_id: int | None = Query(default=None),
) -> list[ClaimEntry]:
    sql = """
    SELECT
        id_acm_claims,
        customer_id,
        subject,
        body,
        name,
        phone,
        email,
        status,
        date_insertion
    FROM acm_claims
    """

    params: tuple[Any, ...] | None = None
    if customer_id is not None:
        sql += " WHERE customer_id = %s"
        params = (customer_id,)

    try:
        rows = await run_in_threadpool(_fetch_all, sql, params)
        return [ClaimEntry(**row) for row in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc


@app.post("/api/call/initiate", response_model=CallInitiateResponse)
async def api_initiate_call(
    payload: CallInitiateRequest = Body(...),
) -> CallInitiateResponse:
    info = await run_in_threadpool(get_client_info, payload.customer_id)

    if not info.get("found"):
        if "error" in info:
            raise HTTPException(status_code=500, detail=f"DB error: {info['error']}")
        raise HTTPException(status_code=404, detail="Customer not found")

    from db.tools import get_call_status

    call_status_info = await run_in_threadpool(get_call_status, payload.customer_id)

    if not call_status_info.get("can_call", True):
        raise HTTPException(
            status_code=409,
            detail={
                "blocked": True,
                "status": call_status_info.get("status"),
                "reason": call_status_info.get("reason"),
                "next_call_date": call_status_info.get("next_call_date"),
                "days_left": call_status_info.get("days_left"),
            },
        )

    session_id = str(uuid.uuid4())
    call_id = str(uuid.uuid4())

    active_calls[call_id] = {
        "customer_id": payload.customer_id,
        "status": "ringing",
        "agent_ws": None,
        "client_ws": None,
        "transcripts": [],
    }

    # Mark IN_CALL here (call is initiated) rather than in start_ws().
    try:
        from db.tools import set_call_status

        await run_in_threadpool(
            set_call_status,
            payload.customer_id,
            "IN_CALL",
            None,
            session_id,
            "Appel initié",
        )
    except Exception:
        pass

    # TODO Week 2 — start VoiceAgent session via WebRTC/WebSocket
    return CallInitiateResponse(
        session_id=session_id,
        call_id=call_id,
        customer_id=payload.customer_id,
        status="initiated",
        customer_name=info.get("customer_name"),
    )


@app.get("/api/call/{call_id}/status", response_model=CallStatusResponse)
async def api_get_call_status(call_id: str) -> CallStatusResponse:
    call = active_calls.get(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")

    return CallStatusResponse(
        call_id=call_id,
        status=call.get("status", "unknown"),
        customer_id=call.get("customer_id"),
    )


@app.post("/api/call/{call_id}/reject", response_model=CallStatusResponse)
async def api_reject_call(call_id: str) -> CallStatusResponse:
    call = active_calls.get(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")

    call["status"] = "rejected"

    customer_id = call.get("customer_id")
    if customer_id:
        await _free_call_status_db(customer_id, "Appel rejeté depuis dashboard")

    agent_ws = call.get("agent_ws")
    if agent_ws is not None:
        try:
            await agent_ws.send_text(json.dumps({"type": "call_rejected"}))
        except Exception:
            pass

    return CallStatusResponse(
        call_id=call_id,
        status=call.get("status", "rejected"),
        customer_id=call.get("customer_id"),
    )


@app.post("/api/mcp/tool")
async def call_mcp_tool_endpoint(
    tool_name: str = Body(..., embed=True),
    arguments: dict = Body(..., embed=True),
) -> dict:
    """
    HTTP proxy to MCP Server.
    Allows Angular dashboard to call MCP tools directly.
    Example: POST /api/mcp/tool {"tool_name": "get_client_info", "arguments": {"customer_id": 1001}}
    """
    from llm.mcp_client import ACMMCPClient

    try:
        mcp = ACMMCPClient.get_instance()
        result = await run_in_threadpool(mcp.call_tool, tool_name, arguments)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MCP error: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
