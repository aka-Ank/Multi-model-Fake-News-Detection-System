from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_root = root / "images"
        self.image_root.mkdir(exist_ok=True)
        self.path = root / "research.sqlite3"
        self._lock = threading.RLock()
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT,
                    progress_stage TEXT, result_json TEXT, error TEXT, retry_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY, submitted_url TEXT NOT NULL, canonical_url TEXT,
                    headline TEXT, body TEXT NOT NULL, publisher TEXT,
                    published_at TEXT, extracted_at TEXT NOT NULL, content_hash TEXT NOT NULL,
                    warnings_json TEXT NOT NULL, images_json TEXT NOT NULL,
                    page_title TEXT, is_live_blog INTEGER NOT NULL DEFAULT 0, selected_segment_json TEXT
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY, input_type TEXT NOT NULL, article_id TEXT,
                    exact_text TEXT NOT NULL, image_key TEXT, image_source_url TEXT, image_sha256 TEXT,
                    image_width INTEGER, image_height INTEGER, image_format TEXT,
                    created_at TEXT NOT NULL, status TEXT NOT NULL,
                    predictions_json TEXT, warnings_json TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY(article_id) REFERENCES articles(id)
                );
                CREATE INDEX IF NOT EXISTS analyses_created ON analyses(created_at DESC);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(analyses)")}
            if "image_sha256" not in columns:
                db.execute("ALTER TABLE analyses ADD COLUMN image_sha256 TEXT")
            for column, sql_type in (("image_width", "INTEGER"), ("image_height", "INTEGER"), ("image_format", "TEXT")):
                if column not in columns:
                    db.execute(f"ALTER TABLE analyses ADD COLUMN {column} {sql_type}")
            job_columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "progress_stage" not in job_columns:
                db.execute("ALTER TABLE jobs ADD COLUMN progress_stage TEXT")
            article_columns = {row[1] for row in db.execute("PRAGMA table_info(articles)")}
            for column, sql_type in (("page_title", "TEXT"), ("is_live_blog", "INTEGER NOT NULL DEFAULT 0"), ("selected_segment_json", "TEXT")):
                if column not in article_columns:
                    db.execute(f"ALTER TABLE articles ADD COLUMN {column} {sql_type}")
            db.execute("UPDATE jobs SET status='failed',progress_stage='interrupted',completed_at=?,error='Service restarted before this job completed.' WHERE status IN ('queued','running')", (now(),))
            db.execute("UPDATE analyses SET status='failed' WHERE status IN ('queued','running')")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        return db

    def create_job(self, kind: str) -> str:
        job_id = str(uuid.uuid4())
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO jobs(id,kind,status,created_at,progress_stage) VALUES(?,?,?,?,?)", (job_id, kind, "queued", now(), "queued"))
        return job_id

    def set_job(self, job_id: str, *, status: str, result: Any = None, error: str | None = None) -> None:
        with self._lock, self._connect() as db:
            if status == "running":
                db.execute("UPDATE jobs SET status=?,progress_stage=?,started_at=? WHERE id=?", (status, status, now(), job_id))
            else:
                db.execute("UPDATE jobs SET status=?,progress_stage=?,completed_at=?,result_json=?,error=? WHERE id=?", (status, status, now(), json.dumps(result) if result is not None else None, error, job_id))

    def set_analysis_status(self, analysis_id: str, status: str) -> None:
        with self._lock, self._connect() as db:
            db.execute("UPDATE analyses SET status=? WHERE id=?", (status, analysis_id))

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["result"] = json.loads(item.pop("result_json")) if item.get("result_json") else None
        return item

    def save_article(self, article: dict[str, Any]) -> str:
        article_id = str(uuid.uuid4())
        with self._lock, self._connect() as db:
            db.execute("""INSERT INTO articles(id,submitted_url,canonical_url,headline,body,publisher,published_at,extracted_at,content_hash,warnings_json,images_json,page_title,is_live_blog,selected_segment_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                article_id, article["submitted_url"], article.get("canonical_url"), article.get("headline"),
                article.get("body", ""), article.get("publisher"), article.get("published_at"), now(),
                article["content_hash"], json.dumps(article.get("warnings", [])), json.dumps(article.get("images", [])),
                article.get("page_title"), int(bool(article.get("is_live_blog"))), json.dumps(article.get("selected_segment")),
            ))
        return article_id

    def get_article(self, article_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM articles WHERE id=?", (article_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["warnings"] = json.loads(result.pop("warnings_json"))
        result["images"] = json.loads(result.pop("images_json"))
        result["is_live_blog"] = bool(result["is_live_blog"])
        result["selected_segment"] = json.loads(result.pop("selected_segment_json")) if result.get("selected_segment_json") else None
        return result

    def create_analysis(self, input_type: str, text: str, image_key: str | None, image_source_url: str | None,
                        image_sha256: str | None, image_width: int | None, image_height: int | None,
                        image_format: str | None, article_id: str | None) -> str:
        analysis_id = str(uuid.uuid4())
        with self._lock, self._connect() as db:
            db.execute("""INSERT INTO analyses(id,input_type,article_id,exact_text,image_key,image_source_url,image_sha256,image_width,image_height,image_format,created_at,status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (analysis_id, input_type, article_id, text, image_key, image_source_url, image_sha256, image_width, image_height, image_format, now(), "queued"))
        return analysis_id

    def finish_analysis(self, analysis_id: str, predictions: list[dict[str, Any]]) -> None:
        states = {item.get("status") for item in predictions}
        status = "completed" if "completed" in states and not (states - {"completed", "skipped"}) else ("partially_completed" if "completed" in states else "failed")
        with self._lock, self._connect() as db:
            db.execute("UPDATE analyses SET status=?,predictions_json=? WHERE id=?", (status, json.dumps(predictions), analysis_id))

    def list_analyses(self, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        page = max(page, 1)
        page_size = min(max(page_size, 1), 100)
        with self._connect() as db:
            total = db.execute("SELECT count(*) FROM analyses").fetchone()[0]
            rows = db.execute("SELECT * FROM analyses ORDER BY created_at DESC LIMIT ? OFFSET ?", (page_size, (page - 1) * page_size)).fetchall()
        return {"items": [self._analysis(row) for row in rows], "page": page, "page_size": page_size, "total": total}

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        return self._analysis(row) if row else None

    @staticmethod
    def _analysis(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["predictions"] = json.loads(item.pop("predictions_json")) if item.get("predictions_json") else None
        item["warnings"] = json.loads(item.pop("warnings_json"))
        return item
