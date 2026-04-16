from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional


SESSION_TTL_DAYS = 14


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"{salt.hex()}:{derived.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split(":", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return hmac.compare_digest(expected, actual)


@dataclass
class UserRecord:
    id: int
    email: str
    full_name: str
    auth_provider: str
    google_sub: Optional[str]
    avatar_url: Optional[str]

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "auth_provider": self.auth_provider,
            "avatar_url": self.avatar_url,
        }


class AuthStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    full_name TEXT NOT NULL,
                    password_hash TEXT,
                    auth_provider TEXT NOT NULL,
                    google_sub TEXT UNIQUE,
                    avatar_url TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );
                """
            )

    def create_local_user(self, *, email: str, full_name: str, password: str) -> UserRecord:
        password_hash = hash_password(password)
        created_at = _to_iso(utcnow())
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (email, full_name, password_hash, auth_provider, created_at)
                VALUES (?, ?, ?, 'local', ?)
                """,
                (email.lower().strip(), full_name.strip(), password_hash, created_at),
            )
            user_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT id, email, full_name, auth_provider, google_sub, avatar_url FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return self._row_to_user(row)

    def authenticate_local_user(self, *, email: str, password: str) -> Optional[UserRecord]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, email, full_name, auth_provider, google_sub, avatar_url, password_hash
                FROM users
                WHERE email = ?
                """,
                (email.lower().strip(),),
            ).fetchone()
        if row is None or row["password_hash"] is None:
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        return UserRecord(
            id=row["id"],
            email=row["email"],
            full_name=row["full_name"],
            auth_provider=row["auth_provider"],
            google_sub=row["google_sub"],
            avatar_url=row["avatar_url"],
        )

    def upsert_google_user(
        self,
        *,
        email: str,
        full_name: str,
        google_sub: str,
        avatar_url: Optional[str],
    ) -> UserRecord:
        normalized_email = email.lower().strip()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, email, full_name, auth_provider, google_sub, avatar_url
                FROM users
                WHERE google_sub = ? OR email = ?
                """,
                (google_sub, normalized_email),
            ).fetchone()
            if row is None:
                cursor = connection.execute(
                    """
                    INSERT INTO users (email, full_name, password_hash, auth_provider, google_sub, avatar_url, created_at)
                    VALUES (?, ?, NULL, 'google', ?, ?, ?)
                    """,
                    (normalized_email, full_name.strip(), google_sub, avatar_url, _to_iso(utcnow())),
                )
                row = connection.execute(
                    "SELECT id, email, full_name, auth_provider, google_sub, avatar_url FROM users WHERE id = ?",
                    (int(cursor.lastrowid),),
                ).fetchone()
            else:
                connection.execute(
                    """
                    UPDATE users
                    SET email = ?, full_name = ?, auth_provider = 'google', google_sub = ?, avatar_url = ?
                    WHERE id = ?
                    """,
                    (normalized_email, full_name.strip(), google_sub, avatar_url, row["id"]),
                )
                row = connection.execute(
                    "SELECT id, email, full_name, auth_provider, google_sub, avatar_url FROM users WHERE id = ?",
                    (row["id"],),
                ).fetchone()
        return self._row_to_user(row)

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        created_at = utcnow()
        expires_at = created_at + timedelta(days=SESSION_TTL_DAYS)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (token, user_id, _to_iso(created_at), _to_iso(expires_at)),
            )
        return token

    def delete_session(self, token: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token = ?", (token,))

    def get_user_by_session(self, token: str | None) -> Optional[UserRecord]:
        if not token:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT s.expires_at, u.id, u.email, u.full_name, u.auth_provider, u.google_sub, u.avatar_url
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token = ?
                """,
                (token,),
            ).fetchone()
            if row is None:
                return None
            if _from_iso(row["expires_at"]) < utcnow():
                connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
                return None
        return UserRecord(
            id=row["id"],
            email=row["email"],
            full_name=row["full_name"],
            auth_provider=row["auth_provider"],
            google_sub=row["google_sub"],
            avatar_url=row["avatar_url"],
        )

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> UserRecord:
        return UserRecord(
            id=row["id"],
            email=row["email"],
            full_name=row["full_name"],
            auth_provider=row["auth_provider"],
            google_sub=row["google_sub"],
            avatar_url=row["avatar_url"],
        )
