"""HTTP authentication, owner isolation, and durable control integration."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def wait_for(studio, endpoint, token, status, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = studio.json(endpoint, token=token)
        assert job["status"] != "failed", job["error"]
        if job["status"] == status:
            return job
        time.sleep(1)
    raise AssertionError(f"Book did not reach {status} within {timeout}s")


def test_registration_login_and_input_boundaries(studio):
    credentials = studio.credentials()
    for key in (None, "wrong-invite", "é"):
        headers = {"X-Registration-Key": key} if key is not None else {}
        studio.call("/auth/register", method="POST", payload=credentials, headers=headers, expected=403)
    user = studio.register(credentials)
    studio.call("/auth/register", method="POST", payload=credentials,
                headers={"X-Registration-Key": studio.registration_key}, expected=409)
    token = studio.token(credentials)
    assert studio.json("/auth/me", token=token)["id"] == user["id"]
    studio.call("/auth/token", method="POST", form={**credentials, "password": "wrong"}, expected=401)
    studio.call("/auth/me", token="invalid", expected=401)
    studio.call("/jobs", expected=401)
    studio.call("/jobs?limit=101", token=token, expected=422)
    for payload in ({"brief": " "}, {"brief": "x" * 20001}, {"brief": 42}, {"prompts": {}}):
        studio.call("/jobs", method="POST", token=token, payload=payload, expected=422)
    studio.call("/auth/register", method="POST", payload={"username": "!", "password": "short"},
                headers={"X-Registration-Key": studio.registration_key}, expected=422)


def test_job_ownership_pause_resume_and_review_validation(studio, accounts):
    owner, other = accounts
    job = studio.json("/jobs", method="POST", token=owner["token"], expected=202)
    endpoint = "/jobs/" + job["id"]
    for suffix in ("", "/artifacts/book.pdf", "/artifacts/chapter-1.md"):
        studio.call(endpoint + suffix, token=other["token"], expected=404)
    for action in ("pause", "resume"):
        studio.call(endpoint + "/" + action, method="POST", token=other["token"], expected=404)
    studio.call(endpoint + "/review", method="POST", token=other["token"], payload={"approved": True}, expected=404)
    studio.call(endpoint + "/pause", method="POST", token=owner["token"], expected=202)
    wait_for(studio, endpoint, owner["token"], "paused")
    studio.call(endpoint + "/artifacts/book.pdf", token=owner["token"], expected=404)
    studio.call(endpoint + "/artifacts/not-allowed.md", token=owner["token"], expected=404)
    for payload in ({"approved": False}, {"approved": False, "feedback": " "}, {"approved": True, "feedback": "no"}, {"approved": False, "feedback": "x" * 2001}):
        studio.call(endpoint + "/review", method="POST", token=owner["token"], payload=payload, expected=422)
    studio.call(endpoint + "/review", method="POST", token=owner["token"], payload={"approved": True}, expected=409)
    studio.call(endpoint + "/resume", method="POST", token=owner["token"], expected=202)
    studio.call(endpoint + "/pause", method="POST", token=owner["token"], expected=202)
    wait_for(studio, endpoint, owner["token"], "paused")
    assert any(item["id"] == job["id"] for item in studio.json("/jobs", token=owner["token"]))
    assert not any(item["id"] == job["id"] for item in studio.json("/jobs", token=other["token"]))


def test_frontend_static_security_boundary(studio):
    html, headers = studio.call("/")
    assert b"Book Writer" in html
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"
    for asset in ("app.css", "app.js"):
        assert len(studio.call("/static/" + asset)[0]) > 100
    for path in ("/static/book_studio/config.py", "/static/.env", "/static/../README.md"):
        studio.call(path, expected=404)


def test_cli_rejects_invalid_prompt_configuration_before_output(tmp_path):
    env_file = tmp_path / "cli.env"
    env_file.write_text("VLLM_BASE_URL=https://example.com/v1\nVLLM_API_KEY=unused\nTAVILY_API_KEY=unused\n")
    overrides = tmp_path / "prompts.json"
    output = tmp_path / "book.md"
    cases = [
        ({"profiles": {"general": {"prompts": {"unknown_agent": {}}}}}, "unknown fields"),
        ({"profiles": {"upi": {"prompts": {"writer": {"human": "No source context"}}}}}, "placeholders"),
        ({"profiles": {"general": {"research": {"seeds": [[["https://bad%.example", "Bad"]], [], []]}}}}, "permitted HTTPS URL"),
        ({"profiles": {"general": {"research": {"seeds": [[["https://127.0.0.1", "Private"]], [], []]}}}}, "permitted HTTPS URL"),
    ]
    # Exercise the documented CLI/config boundary; invalid input must never call providers.
    env = {key: value for key, value in os.environ.items() if key not in (
        "VLLM_BASE_URL", "VLLM_API_KEY", "TAVILY_API_KEY", "BOOK_PROMPTS_FILE"
    )}
    for payload, expected in cases:
        overrides.write_text(json.dumps(payload))
        result = subprocess.run([
            sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "book_writer.py"),
            "--env-file", str(env_file), "--prompts-file", str(overrides), "--output", str(output),
        ], env=env, capture_output=True, text=True, timeout=25)
        assert result.returncode != 0 and expected in result.stderr
        assert not output.exists(), "Invalid prompt settings produced a book artifact"
