import json
import time

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from nanotron_control.auth.audit import AuditLogger
from nanotron_control.auth.jwt import JWTAuth, decode_token, encode_token
from nanotron_control.auth.rbac import Role, requires_role


def test_encode_decode_round_trip():
    token = encode_token({"sub": "ricky", "role": "ADMIN"}, secret="s3cret", ttl_s=10)
    payload = decode_token(token, "s3cret")
    assert payload["sub"] == "ricky"
    assert payload["role"] == "ADMIN"
    assert payload["exp"] > time.time()


def test_decode_rejects_bad_signature():
    token = encode_token({"sub": "x"}, secret="aaa")
    with pytest.raises(ValueError):
        decode_token(token, "bbb")


def test_decode_rejects_expired_token():
    token = encode_token({"sub": "x"}, secret="aaa", ttl_s=-1)
    with pytest.raises(ValueError, match="expired"):
        decode_token(token, "aaa")


def test_role_parse_caseinsensitive():
    assert Role.parse("admin") == Role.ADMIN
    assert Role.parse("trader") == Role.TRADER


def test_rbac_blocks_insufficient_role(monkeypatch):
    auth = JWTAuth(secret="s")
    app = FastAPI()

    @app.get("/admin")
    async def admin_only(claims: dict = Depends(requires_role(Role.ADMIN, auth))):
        return claims

    @app.get("/trader")
    async def trader_only(claims: dict = Depends(requires_role(Role.TRADER, auth))):
        return claims

    viewer_token = encode_token({"sub": "v", "role": "VIEWER"}, secret="s")
    trader_token = encode_token({"sub": "t", "role": "TRADER"}, secret="s")
    admin_token = encode_token({"sub": "a", "role": "ADMIN"}, secret="s")

    with TestClient(app) as c:
        # viewer hits trader → 403
        r = c.get("/trader", headers={"Authorization": f"Bearer {viewer_token}"})
        assert r.status_code == 403

        # trader hits trader → 200
        r = c.get("/trader", headers={"Authorization": f"Bearer {trader_token}"})
        assert r.status_code == 200

        # trader hits admin → 403
        r = c.get("/admin", headers={"Authorization": f"Bearer {trader_token}"})
        assert r.status_code == 403

        # admin hits admin → 200
        r = c.get("/admin", headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200


def test_rbac_returns_401_without_bearer():
    auth = JWTAuth(secret="s")
    app = FastAPI()

    @app.get("/x")
    async def handler(claims: dict = Depends(requires_role(Role.VIEWER, auth))):
        return claims

    with TestClient(app) as c:
        assert c.get("/x").status_code == 401


@pytest.mark.asyncio
async def test_audit_logger_writes_ndjson(tmp_path):
    log = tmp_path / "audit.ndjson"
    al = AuditLogger(log_path=log)
    await al.emit(
        actor="ricky",
        action="strategy.transition",
        resource="alpha",
        detail={"target": "start"},
    )
    line = log.read_text().strip()
    payload = json.loads(line)
    assert payload["actor"] == "ricky"
    assert payload["resource"] == "alpha"
    assert payload["detail"]["target"] == "start"
