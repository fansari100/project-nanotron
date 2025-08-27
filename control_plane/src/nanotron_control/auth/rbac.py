"""Role-based access control.

Three roles, ordered by privilege:

    VIEWER < TRADER < ADMIN

A VIEWER can read everything; a TRADER can flip strategy state and
edit risk limits; an ADMIN can do anything (including managing other
users in a future commit).  ``requires_role`` is a FastAPI dependency
factory.
"""

from __future__ import annotations

from enum import IntEnum

from fastapi import Depends, HTTPException, status

from .jwt import JWTAuth


class Role(IntEnum):
    VIEWER = 1
    TRADER = 2
    ADMIN = 3

    @classmethod
    def parse(cls, name: str) -> "Role":
        try:
            return cls[name.upper()]
        except KeyError as e:
            raise ValueError(f"unknown role: {name}") from e


def requires_role(min_role: Role, jwt_auth: JWTAuth):
    async def _dep(claims: dict = Depends(jwt_auth)) -> dict:
        try:
            actual = Role.parse(claims.get("role", "VIEWER"))
        except ValueError as e:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(e))
        if actual < min_role:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"need {min_role.name}, have {actual.name}",
            )
        return claims

    return _dep
