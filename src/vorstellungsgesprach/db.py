"""SQLite storage for chat history and user feedback (monitoring)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path("vorstellungsgesprach.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Cria as tabelas se ainda não existirem. Chame isso uma vez no início do app."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                model TEXT,
                sources_json TEXT,
                feedback TEXT
            )
            """
        )
        conn.commit()


def log_interaction(query: str, answer: str, model: str | None, sources: list[dict]) -> int:
    """Salva uma pergunta+resposta e retorna o id gerado (usado depois pra registrar feedback)."""
    import json

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO interactions (timestamp, query, answer, model, sources_json, feedback)
            VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                query,
                answer,
                model,
                json.dumps(sources, ensure_ascii=False),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def set_feedback(interaction_id: int, feedback: str) -> None:
    """feedback deve ser 'up' ou 'down'."""
    with _connect() as conn:
        conn.execute(
            "UPDATE interactions SET feedback = ? WHERE id = ?",
            (feedback, interaction_id),
        )
        conn.commit()


def get_all_interactions() -> list[dict[str, Any]]:
    """Retorna todo o histórico, útil para o dashboard de monitoramento."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM interactions ORDER BY timestamp DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def get_stats() -> dict[str, Any]:
    """Estatísticas simples para o dashboard."""
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        thumbs_up = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE feedback = 'up'"
        ).fetchone()[0]
        thumbs_down = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE feedback = 'down'"
        ).fetchone()[0]
        return {
            "total_interactions": total,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
        }