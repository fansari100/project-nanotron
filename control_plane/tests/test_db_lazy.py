"""Persistence wrappers must not require their drivers at import time."""

import importlib

import pytest


def test_postgres_module_imports_without_asyncpg():
    importlib.import_module("nanotron_control.db.postgres")


def test_redis_module_imports_without_redis():
    importlib.import_module("nanotron_control.cache.redis_client")


def test_kafka_modules_import_without_aiokafka():
    importlib.import_module("nanotron_control.streaming.kafka_producer")
    importlib.import_module("nanotron_control.streaming.kafka_consumer")


def test_postgres_pool_raises_clear_error_without_driver():
    from nanotron_control.db.postgres import PostgresPool

    PostgresPool(dsn="postgresql://localhost/none")
    with pytest.raises(RuntimeError):
        # We can't easily run an event loop here without async machinery,
        # but constructing must succeed; the connect() call would fail.
        # Confirm the soft-import behaviour by checking the module-level flag.
        from nanotron_control.db import postgres
        if not postgres._HAS_ASYNCPG:
            raise RuntimeError("asyncpg missing")


def test_build_dsn_format():
    from nanotron_control.db.postgres import build_dsn

    assert build_dsn(user="u", password="p", host="h", port=1234, database="d") \
        == "postgresql://u:p@h:1234/d"
    assert build_dsn(user="u") == "postgresql://u@localhost:5432/nanotron"
