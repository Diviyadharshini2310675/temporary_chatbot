"""
MySQL database connection management.

Provides a simple connection factory using mysql-connector-python.
Each call to get_conn() returns a fresh connection — pooling is
handled by MySQL Connector's built-in pooling.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import mysql.connector
import mysql.connector.pooling

from app.config import settings

logger = logging.getLogger(__name__)

_pool: mysql.connector.pooling.MySQLConnectionPool | None = None


def get_pool() -> mysql.connector.pooling.MySQLConnectionPool:
    """Return the MySQL connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        logger.info("Creating MySQL connection pool …")
        _pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="spiritual_pool",
            pool_size=10,
            host=settings.mysql_host,
            port=settings.mysql_port,
            database=settings.mysql_dbname,
            user=settings.mysql_user,
            password=settings.mysql_password,
            autocommit=False,
        )
        logger.info(
            "MySQL pool ready (host=%s, db=%s).",
            settings.mysql_host,
            settings.mysql_dbname,
        )
    return _pool


@contextmanager
def get_conn() -> Generator:
    """Context manager that yields a MySQL connection from the pool."""
    pool = get_pool()
    conn = pool.get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def close_pool() -> None:
    """Close the pool (no-op for MySQL connector — connections auto-close)."""
    global _pool
    _pool = None
    logger.info("MySQL connection pool cleared.")
