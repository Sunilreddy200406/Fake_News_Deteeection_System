from __future__ import annotations

import csv
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "history.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS verification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                news_summary TEXT NOT NULL,
                source_url TEXT,
                result TEXT NOT NULL,
                method TEXT NOT NULL
            )
            """
        )
        conn.commit()


def summarize_text(text: str, max_len: int = 140) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def save_history(news_text: str, source_url: str, result: str, method: str) -> None:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = summarize_text(news_text)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO verification_history (created_at, news_summary, source_url, result, method)
            VALUES (?, ?, ?, ?, ?)
            """,
            (created_at, summary, source_url or None, result, method),
        )
        conn.commit()


def fetch_history(limit: int = 50, offset: int = 0, result_filter: Optional[str] = None) -> List[Dict[str, str]]:
    query = """
        SELECT id, created_at, news_summary, source_url, result, method
        FROM verification_history
    """
    params = []
    if result_filter:
        query += " WHERE result = ?"
        params.append(result_filter)
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def clear_history() -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM verification_history")
        conn.commit()


def export_history_to_csv(csv_path: Path) -> Path:
    rows = fetch_history(limit=1_000_000, offset=0, result_filter=None)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "News Summary", "Source", "Result", "Method"])
        for row in rows:
            writer.writerow(
                [
                    row["created_at"],
                    row["news_summary"],
                    row["source_url"] or "",
                    row["result"],
                    row["method"],
                ]
            )
    return csv_path

