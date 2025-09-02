"""HS256 JWT helpers + a FastAPI dependency.

Pure stdlib (hmac + json + base64), no external JWT library.  HS256
only — for asymmetric (RS256, EdDSA) the recommendation is to mount a
proper IdP (Auth0 / Keycloak / Okta) in front of the API and accept
its bearer tokens here.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from fastapi import Header, HTTPException, status


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    rem = len(data) % 4
    if rem:
        data += "=" * (4 - rem)
    return base64.urlsafe_b64decode(data)


def encode_token(payload: dict, secret: str, ttl_s: int | None = 3600) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    body = dict(payload)
    now = int(time.time())
    body.setdefault("iat", now)
    if ttl_s is not None:
        body["exp"] = now + ttl_s
    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url_encode(json.dumps(body, separators=(",", ":")).encode())
    msg = f"{h}.{p}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url_encode(sig)}"


def decode_token(token: str, secret: str) -> dict:
    try:
        h, p, s = token.split(".")
    except ValueError as e:
        raise ValueError("malformed jwt") from e
    msg = f"{h}.{p}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url_decode(s), sig):
        raise ValueError("bad signature")
    payload = json.loads(_b64url_decode(p))
    if "exp" in payload and time.time() > payload["exp"]:
        raise ValueError("expired")
    return payload


@dataclass(frozen=True)
class JWTAuth:
    secret: str

    async def __call__(
        self, authorization: str = Header(default="")
    ) -> dict:
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
        token = authorization[len(prefix) :].strip()
        try:
            return decode_token(token, self.secret)
        except ValueError as e:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e)) from e
