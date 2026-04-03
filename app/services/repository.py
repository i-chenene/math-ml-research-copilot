import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models.schemas import DocumentOut


class SQLiteRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                );
                """
            )

    def create_document(self, name: str, source_type: str, source_ref: str | None = None) -> int:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO documents (name, source_type, source_ref, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, source_type, source_ref, created_at),
            )
            return int(cursor.lastrowid)

    def create_chunks(
        self,
        rows: Sequence[tuple[int, int | None, int, str]],
    ) -> list[int]:
        with self._connect() as conn:
            ids: list[int] = []
            for row in rows:
                cursor = conn.execute(
                    """
                    INSERT INTO chunks (document_id, page, chunk_index, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    row,
                )
                ids.append(int(cursor.lastrowid))
            return ids

    def list_documents(self) -> list[DocumentOut]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    d.id,
                    d.name,
                    d.source_type,
                    d.source_ref,
                    d.created_at,
                    COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            ).fetchall()
        return [
            DocumentOut(
                id=int(row["id"]),
                name=row["name"],
                source_type=row["source_type"],
                source_ref=row["source_ref"],
                chunk_count=int(row["chunk_count"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def get_document(self, document_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return dict(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: Sequence[int]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        query = f"""
            SELECT
                c.id AS chunk_id,
                c.document_id AS document_id,
                d.name AS document_name,
                c.page AS page,
                c.chunk_index AS chunk_index,
                c.text AS text
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
        """
        with self._connect() as conn:
            rows = conn.execute(query, tuple(chunk_ids)).fetchall()
        by_id = {int(row["chunk_id"]): dict(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def get_chunks_for_document(self, document_id: int, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id AS chunk_id, page, chunk_index, text
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                LIMIT ?
                """,
                (document_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_chunks(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, text FROM chunks ORDER BY id ASC").fetchall()
        return [dict(row) for row in rows]
