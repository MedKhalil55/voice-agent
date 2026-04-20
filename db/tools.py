import logging
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
) -> dict[str, Any]:
    query = """
    INSERT INTO acm_payment_promise
        (customer_id, amount, installments, promised_date, status, created_by)
    VALUES (%s, %s, %s, %s, 'PROMISED', 'VOICE_AGENT')
    RETURNING id
    """

    try:
        with get_connection() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        query, (customer_id, amount, installments, promised_date)
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
) -> dict[str, Any]:
    query = """
    INSERT INTO acm_call_log
        (customer_id, transcript, intent, outcome, agent_decision, created_by)
    VALUES (%s, %s, %s, %s, %s, 'VOICE_AGENT')
    RETURNING id
    """

    try:
        with get_connection() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        query,
                        (customer_id, transcript, intent, outcome, agent_decision),
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