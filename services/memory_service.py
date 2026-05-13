
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .language_service import clean_text


class MemoryStore:
    def __init__(self, db_path: str, max_history_messages: int = 12) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_history_messages = max_history_messages
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _table_columns(self, conn, table: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row["name"] for row in rows}

    def _ensure_turn_cache_columns(self, conn) -> None:
        cols = self._table_columns(conn, "turn_cache")
        if "assistant_answer" not in cols:
            conn.execute("ALTER TABLE turn_cache ADD COLUMN assistant_answer TEXT DEFAULT ''")

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                ON messages(session_id, id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turn_cache (
                    session_id TEXT PRIMARY KEY,
                    user_message TEXT DEFAULT '',
                    assistant_answer TEXT DEFAULT '',
                    file_context TEXT DEFAULT '',
                    image_context TEXT DEFAULT '',
                    image_ocr TEXT DEFAULT '',
                    attachments_json TEXT DEFAULT '[]',
                    endpoint TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    session_id TEXT NOT NULL,
                    pref_key TEXT NOT NULL,
                    pref_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(session_id, pref_key)
                )
                """
            )
            self._ensure_turn_cache_columns(conn)

    def ensure_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, summary)
                VALUES (?, '')
                """,
                (session_id,),
            )

    def get_summary(self, session_id: str) -> str:
        self.ensure_session(session_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return clean_text(row["summary"] if row else "")

    def set_summary(self, session_id: str, summary: str) -> None:
        self.ensure_session(session_id)
        summary = clean_text(summary)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET summary = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (summary, session_id),
            )

    def add_message(self, session_id: str, role: str, content: str) -> None:
        content = clean_text(content)
        if not content:
            return

        self.ensure_session(session_id)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        self.ensure_session(session_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return [{"id": row["id"], "role": row["role"], "content": row["content"]} for row in rows]

    def get_recent_messages(self, session_id: str, limit: int = 12) -> List[Dict[str, str]]:
        rows = self.get_messages(session_id)
        rows = rows[-limit:] if limit > 0 else rows
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def count_messages(self, session_id: str) -> int:
        self.ensure_session(session_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["count"] if row else 0)

    def clear(self, session_id: str) -> None:
        self.ensure_session(session_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE sessions SET summary = '', updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,),
            )

    def get_context(self, session_id: str, keep_last: int = 12) -> Tuple[str, List[Dict[str, str]]]:
        summary = self.get_summary(session_id)
        recent = self.get_recent_messages(session_id, keep_last)
        return summary, recent

    def set_turn_cache(
        self,
        session_id: str,
        *,
        user_message: str = "",
        assistant_answer: str = "",
        file_context: str = "",
        image_context: str = "",
        image_ocr: str = "",
        attachments: Optional[List[Dict[str, Any]]] = None,
        endpoint: str = "",
    ) -> None:
        self.ensure_session(session_id)
        payload = json.dumps(attachments or [], ensure_ascii=False)

        with self._connect() as conn:
            self._ensure_turn_cache_columns(conn)
            conn.execute(
                """
                INSERT INTO turn_cache (
                    session_id, user_message, assistant_answer, file_context, image_context, image_ocr,
                    attachments_json, endpoint, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_message=excluded.user_message,
                    assistant_answer=excluded.assistant_answer,
                    file_context=excluded.file_context,
                    image_context=excluded.image_context,
                    image_ocr=excluded.image_ocr,
                    attachments_json=excluded.attachments_json,
                    endpoint=excluded.endpoint,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    clean_text(user_message),
                    clean_text(assistant_answer),
                    clean_text(file_context),
                    clean_text(image_context),
                    clean_text(image_ocr),
                    payload,
                    endpoint,
                ),
            )

    def get_turn_cache(self, session_id: str) -> Dict[str, Any]:
        self.ensure_session(session_id)
        with self._connect() as conn:
            self._ensure_turn_cache_columns(conn)
            row = conn.execute(
                """
                SELECT user_message, assistant_answer, file_context, image_context, image_ocr,
                       attachments_json, endpoint
                FROM turn_cache
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if not row:
            return {}

        try:
            attachments = json.loads(row["attachments_json"] or "[]")
        except Exception:
            attachments = []

        return {
            "user_message": clean_text(row["user_message"] or ""),
            "assistant_answer": clean_text(row["assistant_answer"] or ""),
            "file_context": clean_text(row["file_context"] or ""),
            "image_context": clean_text(row["image_context"] or ""),
            "image_ocr": clean_text(row["image_ocr"] or ""),
            "attachments": attachments if isinstance(attachments, list) else [],
            "endpoint": clean_text(row["endpoint"] or ""),
        }

    def clear_turn_cache(self, session_id: str) -> None:
        self.ensure_session(session_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM turn_cache WHERE session_id = ?", (session_id,))

    def replace_last_assistant(self, session_id: str, assistant_text: str) -> None:
        assistant_text = clean_text(assistant_text)
        if not assistant_text:
            return

        self.ensure_session(session_id)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id
                FROM messages
                WHERE session_id = ? AND role = 'assistant'
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

            if row:
                conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))

            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, 'assistant', ?)",
                (session_id, assistant_text),
            )

    def compact_if_needed(
        self,
        session_id: str,
        groq_service: Any,
        *,
        language: str = "en",
        keep_last: int = 12,
        trigger_after: int = 28,
    ) -> bool:
        total = self.count_messages(session_id)
        if total <= trigger_after:
            return False

        rows = self.get_messages(session_id)
        if len(rows) <= keep_last:
            return False

        older = rows[:-keep_last]
        if not older:
            return False

        older_blob = "\n".join(f"{row['role'].upper()}: {row['content']}" for row in older)
        current_summary = self.get_summary(session_id)

        try:
            new_summary = groq_service.compress_chat_memory(
                previous_summary=current_summary,
                older_messages=older_blob,
                language=language,
            )
        except Exception:
            return False

        if not new_summary:
            return False

        ids_to_delete = [row["id"] for row in older]
        if not ids_to_delete:
            return False

        with self._connect() as conn:
            placeholders = ",".join(["?"] * len(ids_to_delete))
            conn.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})",
                ids_to_delete,
            )

        self.set_summary(session_id, new_summary)
        return True

    def set_user_preference(self, session_id: str, key: str, value: str) -> None:
        self.ensure_session(session_id)
        key = clean_text(key)
        value = clean_text(value)
        if not key:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences (session_id, pref_key, pref_value, created_at, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id, pref_key) DO UPDATE SET
                    pref_value=excluded.pref_value,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (session_id, key, value),
            )

    def get_user_preference(self, session_id: str, key: str) -> Optional[str]:
        self.ensure_session(session_id)
        key = clean_text(key)
        if not key:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT pref_value
                FROM user_preferences
                WHERE session_id = ? AND pref_key = ?
                """,
                (session_id, key),
            ).fetchone()
        if not row:
            return None
        return clean_text(row["pref_value"] or "") or None

    def set_current_topic(self, session_id: str, topic: str) -> None:
        self.set_user_preference(session_id, "current_topic", topic)

    def get_current_topic(self, session_id: str) -> Optional[str]:
        return self.get_user_preference(session_id, "current_topic")
