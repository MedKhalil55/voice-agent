"""ACM Banking Voice Agent — MCP Server.

Exposes banking tools (client info, payment promises, claims, call logs and
call status) over the Model Context Protocol using the stdio transport.

Run directly with:
    uv run mcp_server/server.py

Or inspect visually with:
    npx @modelcontextprotocol/inspector uv run mcp_server/server.py
"""

import json
import os
import sys

# Ensure the project root is importable when this file is launched directly as
# a subprocess script (e.g. `uv run mcp_server/server.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from psycopg2.extras import RealDictCursor

from db.database import get_connection
from db.tools import (
    create_claim,
    create_payment_promise,
    get_call_status,
    get_client_info,
    log_call,
)

server = Server("acm-banking-voice-agent")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_client_info",
            description="Récupère les informations complètes d'un client bancaire ACM",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                },
                "required": ["customer_id"],
            },
        ),
        Tool(
            name="create_payment_promise",
            description="Enregistre une promesse de paiement acceptée par le client",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "amount": {"type": "number"},
                    "installments": {"type": "integer"},
                    "promised_date": {"type": "string"},
                    "reason": {"type": "string", "default": "other"},
                    "reason_raw": {"type": "string", "default": ""},
                },
                "required": [
                    "customer_id",
                    "amount",
                    "installments",
                    "promised_date",
                ],
            },
        ),
        Tool(
            name="create_claim",
            description="Enregistre une réclamation déposée par le client",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "name": {"type": "string"},
                    "phone": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["customer_id", "subject", "body"],
            },
        ),
        Tool(
            name="log_call",
            description="Enregistre un événement d'appel dans l'historique",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "transcript": {"type": "string"},
                    "intent": {"type": "string"},
                    "outcome": {"type": "string"},
                    "agent_decision": {"type": "string", "default": ""},
                    "session_id": {"type": "string", "default": ""},
                    "turn_number": {"type": "integer", "default": 1},
                },
                "required": ["customer_id", "transcript", "intent", "outcome"],
            },
        ),
        Tool(
            name="get_call_history",
            description="Récupère l'historique des appels d'un client",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["customer_id"],
            },
        ),
        Tool(
            name="get_call_status",
            description="Vérifie si un client peut être appelé et son statut actuel",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                },
                "required": ["customer_id"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------
def _get_call_history(customer_id: int, limit: int = 10) -> dict:
    query = """
    SELECT id, transcript, intent, outcome, agent_decision,
           session_id, turn_number, call_date
    FROM acm_call_log
    WHERE customer_id = %s
    ORDER BY call_date DESC
    LIMIT %s
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (customer_id, limit))
            rows = cur.fetchall()
    return {"found": True, "calls": [dict(r) for r in rows]}


def _serialize(result: object) -> list[TextContent]:
    return [
        TextContent(
            type="text",
            text=json.dumps(result, default=str, ensure_ascii=False),
        )
    ]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    arguments = arguments or {}
    print(
        f"[MCP-SERVER] Tool appelé: {name} | args: {arguments}",
        file=sys.stderr,
        flush=True,
    )
    try:
        if name == "get_client_info":
            result = get_client_info(int(arguments["customer_id"]))

        elif name == "create_payment_promise":
            result = create_payment_promise(
                customer_id=int(arguments["customer_id"]),
                amount=float(arguments["amount"]),
                installments=int(arguments["installments"]),
                promised_date=arguments["promised_date"],
                reason=arguments.get("reason", "other"),
                reason_raw=arguments.get("reason_raw", ""),
            )

        elif name == "create_claim":
            result = create_claim(
                customer_id=int(arguments["customer_id"]),
                subject=arguments["subject"],
                body=arguments["body"],
                name=arguments.get("name", ""),
                phone=arguments.get("phone", ""),
                email=arguments.get("email", ""),
            )

        elif name == "log_call":
            result = log_call(
                customer_id=int(arguments["customer_id"]),
                transcript=arguments["transcript"],
                intent=arguments["intent"],
                outcome=arguments["outcome"],
                agent_decision=arguments.get("agent_decision", ""),
                session_id=arguments.get("session_id", ""),
                turn_number=int(arguments.get("turn_number", 1)),
            )

        elif name == "get_call_history":
            result = _get_call_history(
                customer_id=int(arguments["customer_id"]),
                limit=int(arguments.get("limit", 10)),
            )

        elif name == "get_call_status":
            result = get_call_status(int(arguments["customer_id"]))

        else:
            result = {"error": f"unknown_tool: {name}"}

        print(
            f"[MCP-SERVER] Tool terminé: {name} | result: {result}",
            file=sys.stderr,
            flush=True,
        )

        return _serialize(result)

    except Exception as exc:  # noqa: BLE001
        print(
            f"[MCP-SERVER] Tool erreur: {name} | error: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
