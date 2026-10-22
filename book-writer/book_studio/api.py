"""Authenticated HTTP interface for the book-writing service."""

from __future__ import annotations

import logging
import secrets
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Path as ApiPath, Query, Request
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .auth import DUMMY_HASH, PASSWORD_HASH, access_token, current_user
from .config import STATIC_DIR, load_service_config
from .jobs import ARTIFACT_NAMES, JobConflict, JobNotFound, JobService


class Registration(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    password: str = Field(min_length=12, max_length=128)


class UserPublic(BaseModel):
    id: int
    username: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ReviewRequest(BaseModel):
    approved: bool
    feedback: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def check_feedback(self) -> "ReviewRequest":
        self.feedback = self.feedback.strip()
        if not self.approved and not self.feedback:
            raise ValueError("A rejected chapter needs feedback")
        if self.approved and self.feedback:
            raise ValueError("Feedback is only used when rejecting a chapter")
        return self


class JobSummary(BaseModel):
    id: str
    brief: str
    status: str
    chapter_index: int
    created_at: str
    updated_at: str


class JobDetail(JobSummary):
    review_chapter: int | None
    review_preview: str | None
    error: str | None


class BookRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    brief: str | None = Field(default=None, max_length=20000)

    @model_validator(mode="after")
    def check_brief(self) -> "BookRequest":
        if self.brief is not None:
            self.brief = self.brief.strip()
            if not self.brief:
                raise ValueError("Book brief must not be blank")
        return self


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_service_config()
    jwt_secret = config.get("JWT_SECRET", "")
    registration_key = config.get("REGISTRATION_KEY", "")
    service = JobService(Path(config.get("DATA_DIR", "data")), config)
    app.state.service = service
    app.state.jwt_secret = jwt_secret
    app.state.registration_key = registration_key
    service.start()
    try:
        yield
    finally:
        service.stop()


app = FastAPI(title="Book Writer", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def workspace() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'; form-action 'self'",
    })


@app.exception_handler(JobNotFound)
async def not_found_handler(request: Request, exc: JobNotFound):
    raise HTTPException(status_code=404, detail="Job not found")


@app.exception_handler(JobConflict)
async def conflict_handler(request: Request, exc: JobConflict):
    raise HTTPException(status_code=409, detail=str(exc))


def service_for(request: Request) -> JobService:
    return request.app.state.service


@app.get("/health")
def health(service: Annotated[JobService, Depends(service_for)]) -> dict:
    with service.store.db() as conn:
        conn.execute("SELECT 1").fetchone()
    if not service.healthy():
        raise HTTPException(status_code=503, detail="Job workers are unavailable")
    return {"status": "ok"}


@app.post("/auth/register", response_model=UserPublic, status_code=201)
def register(
    payload: Registration,
    request: Request,
    service: Annotated[JobService, Depends(service_for)],
    registration_key: Annotated[str | None, Header(alias="X-Registration-Key")] = None,
) -> dict:
    if not registration_key or not secrets.compare_digest(
        registration_key.encode(), request.app.state.registration_key.encode()
    ):
        raise HTTPException(status_code=403, detail="Registration is not allowed")
    username = payload.username.lower()
    password_hash = PASSWORD_HASH.hash(payload.password)
    try:
        user_id = service.store.create_user(username, password_hash)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Username already exists") from exc
    return {"id": user_id, "username": username}


@app.post("/auth/token", response_model=TokenResponse)
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    service: Annotated[JobService, Depends(service_for)],
) -> dict:
    user = service.store.user_by_name(form.username.lower())
    if user is None:
        PASSWORD_HASH.verify(form.password, DUMMY_HASH)
    if user is None or not PASSWORD_HASH.verify(form.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    token = access_token(user["id"], request.app.state.jwt_secret)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me", response_model=UserPublic)
def me(user: Annotated[dict, Depends(current_user)]) -> dict:
    return user


@app.post("/jobs", response_model=JobDetail, status_code=202)
def create_job(
    user: Annotated[dict, Depends(current_user)],
    service: Annotated[JobService, Depends(service_for)],
    payload: BookRequest | None = None,
) -> dict:
    brief = payload.brief if payload and payload.brief is not None else service.profiles["default_brief"]
    job_id = service.store.create_job(user["id"], brief)
    return service.store.get_job(user["id"], job_id)


@app.get("/jobs", response_model=list[JobSummary])
def list_jobs(
    user: Annotated[dict, Depends(current_user)],
    service: Annotated[JobService, Depends(service_for)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[dict]:
    return service.store.list_jobs(user["id"], limit)


@app.get("/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: str, user: Annotated[dict, Depends(current_user)],
            service: Annotated[JobService, Depends(service_for)]) -> dict:
    return service.store.get_job(user["id"], job_id)


@app.post("/jobs/{job_id}/pause", response_model=JobDetail, status_code=202)
def pause_job(job_id: str, user: Annotated[dict, Depends(current_user)],
              service: Annotated[JobService, Depends(service_for)]) -> dict:
    service.store.request_pause(user["id"], job_id)
    return service.store.get_job(user["id"], job_id)


@app.post("/jobs/{job_id}/resume", response_model=JobDetail, status_code=202)
def resume_job(job_id: str, user: Annotated[dict, Depends(current_user)],
               service: Annotated[JobService, Depends(service_for)]) -> dict:
    service.store.resume(user["id"], job_id)
    return service.store.get_job(user["id"], job_id)


@app.post("/jobs/{job_id}/review", response_model=JobDetail, status_code=202)
def review_job(job_id: str, payload: ReviewRequest,
               user: Annotated[dict, Depends(current_user)],
               service: Annotated[JobService, Depends(service_for)]) -> dict:
    service.store.submit_review(user["id"], job_id, payload.approved, payload.feedback)
    return service.store.get_job(user["id"], job_id)


@app.get("/jobs/{job_id}/artifacts/{name}")
def download_artifact(
    job_id: str,
    name: Annotated[str, ApiPath(pattern=r"^[A-Za-z0-9.-]+$")],
    user: Annotated[dict, Depends(current_user)],
    service: Annotated[JobService, Depends(service_for)],
) -> FileResponse:
    job = service.store.get_job(user["id"], job_id)
    if name not in ARTIFACT_NAMES or (name.startswith("book.") and job["status"] != "completed"):
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = service.store.artifacts_dir / job_id / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    media_type = "application/pdf" if name.endswith(".pdf") else "text/markdown; charset=utf-8"
    return FileResponse(path, media_type=media_type, filename=name)
