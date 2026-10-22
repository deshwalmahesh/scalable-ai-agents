"""Durable job storage, artifact rendering, and bounded workflow execution."""

from __future__ import annotations

import html
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from .config import load_profiles
from .writer import BookWriter, initial_state, render_book


LOG = logging.getLogger("book_jobs")
ARTIFACT_NAMES = {"book.md", "book.pdf", "chapter-1.md", "chapter-2.md", "chapter-3.md"}


class JobNotFound(Exception):
    pass


class JobConflict(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_book_pdf(markdown: str, output: Path) -> None:
    configured_font = os.environ.get("PDF_FONT_PATH")
    candidates = [configured_font] if configured_font else [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
    ]
    required_glyphs = {ord(char) for char in markdown if char.isprintable()}
    font_name = None
    for font_path in candidates:
        if not font_path or not Path(font_path).is_file():
            continue
        name = "BookFont_" + hashlib.sha256(font_path.encode()).hexdigest()[:8]
        font = TTFont(name, font_path)
        if not required_glyphs.issubset(font.face.charWidths):
            continue
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(font)
        font_name = name
        break
    if font_name is None:
        raise RuntimeError("No PDF font covers every book character; set PDF_FONT_PATH")

    def linked_text(text: str) -> str:
        parts = re.split(r"(<https://[^>]+>)", text.replace("*", ""))
        result = []
        for part in parts:
            if part.startswith("<https://") and part.endswith(">"):
                url = part[1:-1]
                safe = html.escape(url, quote=True)
                result.append(f'<link href="{safe}" color="#0645ad">{safe}</link>')
            else:
                result.append(html.escape(part))
        return "".join(result)

    body = ParagraphStyle("Body", fontName=font_name, fontSize=10, leading=16, spaceAfter=10)
    title = ParagraphStyle("Title", parent=body, fontSize=20, leading=26, spaceAfter=18)
    chapter = ParagraphStyle("Chapter", parent=body, fontSize=15, leading=21, spaceAfter=14, keepWithNext=True)
    subheading = ParagraphStyle("Subheading", parent=body, fontSize=11, leading=16, spaceAfter=8, keepWithNext=True)
    reference = ParagraphStyle("Reference", parent=body, fontSize=8, leading=13, textColor=colors.darkslategrey)
    story = []
    for block in re.split(r"\n\s*\n", markdown.strip()):
        if block.startswith("# "):
            story.append(Paragraph(linked_text(block[2:]), title))
        elif block.startswith("## "):
            if story:
                story.append(PageBreak())
            story.append(Paragraph(linked_text(block[3:]), chapter))
        elif block.startswith("### "):
            story.append(Paragraph(linked_text(block[4:]), subheading))
        elif block.startswith("Takeaway:"):
            story.append(Paragraph(linked_text(block), subheading))
        elif block.startswith("[") and "<https://" in block:
            for line in block.splitlines():
                story.append(Paragraph(linked_text(line), reference))
        else:
            story.append(Paragraph(linked_text(" ".join(block.splitlines())), body))
        story.append(Spacer(1, 3))
    if not story:
        raise ValueError("Cannot render an empty book")
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=52, rightMargin=52,
                      topMargin=55, bottomMargin=55, title="Book").build(story)


class JobStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = data_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "jobs.sqlite3"
        self.checkpoint_path = data_dir / "checkpoints.sqlite3"
        self._init_schema()

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.db() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError(f"Unsupported job database schema version: {version}")
            conn.execute("""CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL, created_at TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
                brief TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN
                ('queued','running','pause_requested','paused','awaiting_review','completed','failed')),
                chapter_index INTEGER NOT NULL DEFAULT 0,
                review_chapter INTEGER, review_checkpoint_id TEXT,
                review_preview TEXT, review_payload TEXT,
                error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_owner_idx ON jobs(user_id, created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs(status, created_at)")
            if version == 0:
                conn.execute("PRAGMA user_version=1")

    def create_user(self, username: str, password_hash: str) -> int:
        with self.db() as conn:
            cursor = conn.execute("INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)",
                                  (username, password_hash, utc_now()))
            return cursor.lastrowid

    def user_by_name(self, username: str) -> dict | None:
        with self.db() as conn:
            row = conn.execute("SELECT id,username,password_hash FROM users WHERE username=?", (username,)).fetchone()
            return dict(row) if row else None

    def user_by_id(self, user_id: int) -> dict | None:
        with self.db() as conn:
            row = conn.execute("SELECT id,username FROM users WHERE id=?", (user_id,)).fetchone()
            return dict(row) if row else None

    def create_job(self, user_id: int, brief: str) -> str:
        job_id = uuid4().hex
        now = utc_now()
        with self.db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','pause_requested')").fetchone()[0]
            own_active = conn.execute("SELECT COUNT(*) FROM jobs WHERE user_id=? AND status IN ('queued','running','pause_requested')", (user_id,)).fetchone()[0]
            if active >= 20 or own_active >= 3:
                raise JobConflict("The job queue is full; retry after a running job finishes")
            conn.execute("INSERT INTO jobs(id,user_id,brief,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                         (job_id, user_id, brief, "queued", now, now))
        return job_id

    def get_job(self, user_id: int, job_id: str) -> dict:
        with self.db() as conn:
            row = conn.execute("""SELECT id,brief,status,chapter_index,review_chapter,review_preview,
                error,created_at,updated_at FROM jobs WHERE id=? AND user_id=?""", (job_id, user_id)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return dict(row)

    def list_jobs(self, user_id: int, limit: int = 50) -> list[dict]:
        with self.db() as conn:
            rows = conn.execute("""SELECT id,brief,status,chapter_index,created_at,updated_at
                FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT ?""", (user_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def _owned_status(self, conn: sqlite3.Connection, user_id: int, job_id: str) -> str:
        row = conn.execute("SELECT status FROM jobs WHERE id=? AND user_id=?", (job_id, user_id)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return row["status"]

    def request_pause(self, user_id: int, job_id: str) -> None:
        with self.db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            status = self._owned_status(conn, user_id, job_id)
            if status in ("queued", "running"):
                target = "paused" if status == "queued" else "pause_requested"
                conn.execute("UPDATE jobs SET status=?,updated_at=? WHERE id=?", (target, utc_now(), job_id))
            elif status not in ("paused", "pause_requested"):
                raise JobConflict(f"Cannot pause a job in {status} state")

    def resume(self, user_id: int, job_id: str) -> None:
        with self.db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            status = self._owned_status(conn, user_id, job_id)
            if status != "paused":
                raise JobConflict(f"Cannot resume a job in {status} state")
            conn.execute("UPDATE jobs SET status='queued',updated_at=? WHERE id=?", (utc_now(), job_id))

    def submit_review(self, user_id: int, job_id: str, approved: bool, feedback: str) -> None:
        with self.db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            status = self._owned_status(conn, user_id, job_id)
            if status != "awaiting_review":
                raise JobConflict(f"Cannot review a job in {status} state")
            row = conn.execute("SELECT review_chapter FROM jobs WHERE id=?", (job_id,)).fetchone()
            payload = json.dumps({"approved": approved, "feedback": feedback})
            conn.execute("""UPDATE jobs SET status='queued',review_payload=?,review_preview=NULL,
                updated_at=? WHERE id=?""", (payload, utc_now(), job_id))
            LOG.info("Review submitted for job %s chapter %s", job_id, row["review_chapter"] + 1)

    def pause_requested(self, job_id: str) -> bool:
        with self.db() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            return bool(row and row["status"] == "pause_requested")

    def recover(self) -> None:
        with self.db() as conn:
            conn.execute("UPDATE jobs SET status='queued',updated_at=? WHERE status='running'", (utc_now(),))
            conn.execute("UPDATE jobs SET status='paused',updated_at=? WHERE status='pause_requested'", (utc_now(),))

    def claim_next(self) -> dict | None:
        with self.db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("""SELECT id,brief,review_chapter,review_checkpoint_id,review_payload FROM jobs
                WHERE status='queued' ORDER BY created_at,id LIMIT 1""").fetchone()
            if row is None:
                return None
            conn.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=?", (utc_now(), row["id"]))
            return dict(row)

    def update_progress(self, job_id: str, chapter_index: int) -> None:
        with self.db() as conn:
            conn.execute("UPDATE jobs SET chapter_index=?,updated_at=? WHERE id=?",
                         (chapter_index, utc_now(), job_id))

    def mark_interrupted(self, job_id: str, kind: str, chapter_index: int,
                         checkpoint_id: str, preview: str | None = None) -> None:
        status = "paused" if kind == "pause" else "awaiting_review"
        with self.db() as conn:
            conn.execute("""UPDATE jobs SET status=?,chapter_index=?,review_chapter=?,
                review_checkpoint_id=?,review_preview=?,review_payload=NULL,updated_at=? WHERE id=?""",
                (status, chapter_index, chapter_index if kind == "review" else None,
                 checkpoint_id if kind == "review" else None, preview, utc_now(), job_id))

    def mark_completed(self, job_id: str) -> None:
        with self.db() as conn:
            conn.execute("""UPDATE jobs SET status='completed',chapter_index=3,
                review_chapter=NULL,review_checkpoint_id=NULL,review_preview=NULL,
                review_payload=NULL,updated_at=? WHERE id=?""",
                (utc_now(), job_id))

    def mark_failed(self, job_id: str, error: str) -> None:
        with self.db() as conn:
            conn.execute("UPDATE jobs SET status='failed',error=?,updated_at=? WHERE id=?",
                         (error[:500], utc_now(), job_id))


class JobService:
    def __init__(self, data_dir: Path, writer_config: dict[str, str], workers: int = 2) -> None:
        self.profiles = load_profiles(writer_config)
        self.store = JobStore(data_dir)
        self.writer_config = writer_config
        self.workers = workers
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        self.store.recover()
        for index in range(self.workers):
            thread = threading.Thread(target=self._worker_loop, name=f"book-worker-{index + 1}", daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=10)
            if thread.is_alive():
                LOG.warning("Worker %s is still finishing an external call", thread.name)

    def healthy(self) -> bool:
        return bool(self.threads) and all(thread.is_alive() for thread in self.threads)

    def _worker_loop(self) -> None:
        writer = BookWriter(self.writer_config, self.profiles)
        while not self.stop_event.is_set():
            job = None
            try:
                job = self.store.claim_next()
                if job is None:
                    self.stop_event.wait(0.25)
                    continue
                LOG.info("Started job %s", job["id"])
                self._run_job(writer, job)
            except Exception as exc:
                if job is not None:
                    LOG.exception("Job %s failed", job["id"])
                    self.store.mark_failed(job["id"], f"{type(exc).__name__}: workflow failed; check server logs")
                else:
                    LOG.exception("Worker could not claim a job")
                    self.stop_event.wait(1)

    def _run_job(self, writer: BookWriter, job: dict) -> None:
        job_id = job["id"]
        conn = sqlite3.connect(self.store.checkpoint_path, timeout=30, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            serde = JsonPlusSerializer(allowed_msgpack_modules=[
                # Preserve checkpoints saved before the backend became a package.
                ("book_writer", "Outline"), ("book_writer", "ChapterPlan"), ("book_writer", "Source"),
                ("book_studio.writer", "Outline"), ("book_studio.writer", "ChapterPlan"),
                ("book_studio.writer", "Source")
            ])
            graph = writer.build_graph(SqliteSaver(conn, serde=serde), self.store.pause_requested, human_review=True)
            config = {"configurable": {"thread_id": job_id}, "recursion_limit": 1000}
            snapshot = graph.get_state(config)
            if snapshot.values:
                self._sync_chapters(job_id, snapshot.values["chapters"])
                self.store.update_progress(job_id, snapshot.values["chapter_index"])
            if snapshot.values and not snapshot.next and len(snapshot.values["chapters"]) == 3:
                self._finish(job_id, snapshot.values)
                return

            pending = [item.value for task in snapshot.tasks for item in task.interrupts]
            if len(pending) > 1:
                raise RuntimeError("Book workflow produced multiple simultaneous interrupts")
            if not snapshot.values:
                graph_input = initial_state(job["brief"], job_id, writer.profile_for(job["brief"]))
            elif pending:
                payload = pending[0]
                if payload["kind"] == "pause":
                    graph_input = Command(resume=True)
                elif payload["kind"] == "review":
                    checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
                    if not job["review_payload"] or job["review_checkpoint_id"] != checkpoint_id:
                        self._record_interrupt(job_id, snapshot, payload)
                        return
                    graph_input = Command(resume=json.loads(job["review_payload"]))
                else:
                    raise RuntimeError("Unknown workflow interrupt")
            else:
                graph_input = None

            for update in graph.stream(graph_input, config, stream_mode="updates"):
                accepted = update.get("accept")
                if accepted:
                    self._sync_chapters(job_id, accepted["chapters"])
                    self.store.update_progress(job_id, accepted["chapter_index"])
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise RuntimeError("Book workflow stopped without state")
            self._sync_chapters(job_id, snapshot.values["chapters"])
            self.store.update_progress(job_id, snapshot.values["chapter_index"])
            pending = [item.value for task in snapshot.tasks for item in task.interrupts]
            if len(pending) == 1:
                self._record_interrupt(job_id, snapshot, pending[0])
            elif not pending and not snapshot.next and len(snapshot.values["chapters"]) == 3:
                self._finish(job_id, snapshot.values)
            else:
                raise RuntimeError("Book workflow stopped before all chapters were accepted")
        finally:
            conn.close()

    def _record_interrupt(self, job_id: str, snapshot, payload: dict) -> None:
        if payload.get("kind") not in ("pause", "review"):
            raise RuntimeError("Unknown workflow interrupt")
        chapter_index = payload["chapter_index"]
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        self.store.mark_interrupted(job_id, payload["kind"], chapter_index,
                                    checkpoint_id, payload.get("chapter"))
        LOG.info("Job %s stopped for %s at chapter %d", job_id, payload["kind"], chapter_index + 1)

    def _atomic_text(self, path: Path, content: str) -> None:
        temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, path)

    def _sync_chapters(self, job_id: str, chapters: list[str]) -> None:
        if not chapters:
            return
        directory = self.store.artifacts_dir / job_id
        directory.mkdir(parents=True, exist_ok=True)
        for number, content in enumerate(chapters, 1):
            path = directory / f"chapter-{number}.md"
            content = content + "\n"
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                self._atomic_text(path, content)

    def _finish(self, job_id: str, state: dict) -> None:
        directory = self.store.artifacts_dir / job_id
        directory.mkdir(parents=True, exist_ok=True)
        book = render_book(state)
        pdf_temp = directory / f".book.{uuid4().hex}.pdf"
        write_book_pdf(book, pdf_temp)
        os.replace(pdf_temp, directory / "book.pdf")
        self._atomic_text(directory / "book.md", book)
        self.store.mark_completed(job_id)
        LOG.info("Completed job %s", job_id)
