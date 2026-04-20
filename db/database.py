import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from dotenv import load_dotenv
from psycopg2 import pool
from psycopg2.extensions import connection

load_dotenv(override=True)

_pool: pool.ThreadedConnectionPool | None = None


def get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            host=os.environ["DB_HOST"],
            port=int(os.environ.get("DB_PORT", 5432)),
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
        )
    return _pool


@contextmanager
def get_connection() -> Iterator[connection]:
    conn = get_pool().getconn()
    try:
        yield conn
    finally:
        get_pool().putconn(conn)
