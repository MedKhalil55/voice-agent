import uuid
from datetime import date, datetime
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import RealDictCursor
from starlette.concurrency import run_in_threadpool

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
    customer_id: int
    status: str
    customer_name: Optional[str] = None


# -------------------------
# Endpoints
# -------------------------


@app.get("/api/clients", response_model=list[ClientSummary])
async def api_get_clients() -> list[ClientSummary]:
    customer_ids = [1001, 1002, 1003]

    results: list[ClientSummary] = []
    for customer_id in customer_ids:
        info = await run_in_threadpool(get_client_info, customer_id)

        if not info.get("found"):
            if "error" in info:
                raise HTTPException(
                    status_code=500, detail=f"DB error: {info['error']}"
                )
            continue
        
        # Calculer le profil
        profile = await run_in_threadpool(classify_client_profile, info)

        results.append(
            ClientSummary(
                customer_id=customer_id,
                customer_name=info.get("customer_name"),
                unpaid_amount=info.get("unpaid_amount"),
                late_days=info.get("late_days"),
                statut_workflow=info.get("statut_workflow"),
                telephone_1=info.get("telephone_1"),
                email=info.get("email"),
                profile_type=profile.get("profile"),  # ← CONTENTIEUX/FIDELE/DIFFICILE
            )
        )

    return results


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

    session_id = str(uuid.uuid4())

    # TODO Week 2 — start VoiceAgent session via WebRTC/WebSocket
    return CallInitiateResponse(
        session_id=session_id,
        customer_id=payload.customer_id,
        status="initiated",
        customer_name=info.get("customer_name"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
