import logging
from datetime import date
from typing import Any

from psycopg2.extras import RealDictCursor

from db.database import get_connection

logger = logging.getLogger(__name__)


def get_client_info(customer_id: int) -> dict[str, Any]:
    query = """
    SELECT
        c.customer_name,
        c.telephone_1,
        c.email,
        c.status as customer_status,
        c.date_de_naissance,
        col.unpaid_amount,
        col.late_days,
        col.number_of_unpaid_installment,
        col.statut_workflow,
        col.account_number,
        l.normal_payment,
        l.apply_amount_total,
        l.term_period
    FROM acm_customer c
    JOIN acm_collection col ON c.customer_id_extern = col.customer_id_extern
    JOIN acm_loan l ON col.id_loan_extern = l.id_loan_extern
    WHERE c.customer_id_extern = %s
      AND col.acm_enabled = TRUE
    LIMIT 1
    """

    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (customer_id,))
                row = cur.fetchone()

        if row is None:
            return {
                "found": False,
                "customer_id": customer_id,
                "message": "Client not found",
            }

        payload = dict(row)
        payload["found"] = True
        return payload
    except Exception as exc:
        logger.exception("Failed to fetch client info for customer_id=%s", customer_id)
        return {
            "found": False,
            "customer_id": customer_id,
            "error": str(exc),
        }


def create_payment_promise(
    customer_id: int,
    amount: float,
    installments: int,
    promised_date: str,
    reason: str = "",  # ← AJOUTER
    reason_raw: str = "",  # ← AJOUTER
) -> dict[str, Any]:
    query = """
    INSERT INTO acm_payment_promise
        (customer_id, amount, installments, promised_date, status, created_by, reason, reason_raw)
    VALUES (%s, %s, %s, %s, 'PROMISED', 'VOICE_AGENT', %s, %s)
    RETURNING id
    """

    try:
        with get_connection() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        query,
                        (
                            customer_id,
                            amount,
                            installments,
                            promised_date,
                            reason,
                            reason_raw,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {
            "success": True,
            "id": row["id"],
            "customer_id": customer_id,
            "status": "PROMISED",
        }
    except Exception as exc:
        logger.exception(
            "Failed to create payment promise for customer_id=%s", customer_id
        )
        return {
            "success": False,
            "customer_id": customer_id,
            "error": str(exc),
        }


def log_call(
    customer_id: int,
    transcript: str,
    intent: str,
    outcome: str,
    agent_decision: str,
    session_id: str = "",
    turn_number: int = 1,
) -> dict[str, Any]:
    query = """
    INSERT INTO acm_call_log
        (
            customer_id,
            transcript,
            intent,
            outcome,
            agent_decision,
            session_id,
            turn_number,
            created_by
        )
    VALUES (%s, %s, %s, %s, %s, %s, %s, 'VOICE_AGENT')
    RETURNING id
    """

    try:
        with get_connection() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        query,
                        (
                            customer_id,
                            transcript,
                            intent,
                            outcome,
                            agent_decision,
                            session_id,
                            turn_number,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {
            "success": True,
            "id": row["id"],
            "customer_id": customer_id,
        }
    except Exception as exc:
        logger.exception("Failed to log call for customer_id=%s", customer_id)
        return {
            "success": False,
            "customer_id": customer_id,
            "error": str(exc),
        }


def create_claim(
    customer_id: int,
    subject: str,
    body: str,
    name: str,
    phone: str,
    email: str,
) -> dict[str, Any]:
    query = """
    INSERT INTO acm_claims
        (
            subject,
            body,
            customer_id,
            name,
            phone,
            email,
            status,
            acm_enabled,
            date_insertion,
            acm_version
        )
    VALUES (%s, %s, %s, %s, %s, %s, 'OPEN', TRUE, CURRENT_DATE, 1)
    RETURNING id_acm_claims
    """

    try:
        with get_connection() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (subject, body, customer_id, name, phone, email))
                    row = cur.fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {
            "success": True,
            "id_acm_claims": row["id_acm_claims"],
            "customer_id": customer_id,
            "status": "OPEN",
        }
    except Exception as exc:
        logger.exception("Failed to create claim for customer_id=%s", customer_id)
        return {
            "success": False,
            "customer_id": customer_id,
            "error": str(exc),
        }


def get_call_status(customer_id: int) -> dict[str, Any]:
    query = """
    SELECT
        customer_id,
        status,
        next_call_date,
        last_call_date,
        session_id,
        notes,
        updated_at
    FROM acm_call_status
    WHERE customer_id = %s
    LIMIT 1
    """

    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (customer_id,))
                row = cur.fetchone()

        if row is None:
            return {
                "can_call": True,
                "status": "FREE",
            }

        status = str(row.get("status") or "FREE").strip().upper()
        next_call_date = row.get("next_call_date")
        today = date.today()

        if status in {"FREE", "BROKEN"}:
            return {
                "can_call": True,
                "status": status,
            }

        if status == "KEPT":
            return {
                "can_call": False,
                "status": "KEPT",
                "reason": "Promesse honorée — dossier clôturé",
            }

        if status == "IN_CALL":
            return {
                "can_call": False,
                "status": "IN_CALL",
                "reason": "Appel en cours",
            }

        if status in {"PROMISED", "CALLBACK", "REFUSED"}:
            if next_call_date is not None and isinstance(next_call_date, date):
                if today < next_call_date:
                    days_left = int((next_call_date - today).days)
                    return {
                        "can_call": False,
                        "status": status,
                        "next_call_date": next_call_date.isoformat(),
                        "days_left": days_left,
                        "reason": f"Prochain appel autorisé dans {days_left} jour(s)",
                    }

            return {
                "can_call": True,
                "status": "BROKEN",
                "reason": "Promesse non honorée — rappel autorisé",
            }

        # Unknown status: be permissive but explicit.
        return {
            "can_call": True,
            "status": status,
        }

    except Exception as exc:
        logger.exception("Failed to fetch call status for customer_id=%s", customer_id)
        return {
            "can_call": True,
            "status": "FREE",
            "error": str(exc),
        }


def set_call_status(
    customer_id: int,
    status: str,
    next_call_date: str | None = None,
    session_id: str = "",
    notes: str = "",
) -> dict[str, Any]:
    query = """
    INSERT INTO acm_call_status
        (customer_id, status, next_call_date, last_call_date, session_id, notes, updated_at)
    VALUES
        (%s, %s, %s, NOW(), %s, %s, NOW())
    ON CONFLICT (customer_id)
    DO UPDATE SET
        status = EXCLUDED.status,
        next_call_date = EXCLUDED.next_call_date,
        last_call_date = NOW(),
        session_id = EXCLUDED.session_id,
        notes = EXCLUDED.notes,
        updated_at = NOW()
    RETURNING id
    """

    parsed_next_call_date: date | None = None
    if next_call_date:
        try:
            parsed_next_call_date = date.fromisoformat(str(next_call_date).strip())
        except Exception as exc:
            raise ValueError(
                f"Invalid next_call_date (expected ISO YYYY-MM-DD): {next_call_date!r}"
            ) from exc

    status_value = str(status or "FREE").strip().upper()

    try:
        with get_connection() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        query,
                        (
                            customer_id,
                            status_value,
                            parsed_next_call_date,
                            session_id,
                            notes,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {
            "success": True,
            "id": (row or {}).get("id"),
            "customer_id": customer_id,
            "status": status_value,
        }
    except Exception as exc:
        logger.exception("Failed to set call status for customer_id=%s", customer_id)
        return {
            "success": False,
            "customer_id": customer_id,
            "error": str(exc),
        }
