"""End-to-end RAG against REAL Qdrant + REAL Gemini (no mocking). Skips without infra.

Covers the TASK's core contract: upload -> ingest -> grounded answer with citations
(document + page), the "no relevant info" refusal, and cross-tenant isolation.
"""

import asyncio
import json
import uuid

from tests.helpers import ASSETS, needs_gemini, needs_qdrant

pytestmark = [needs_gemini, needs_qdrant]


async def _upload_and_wait(client, headers, path, filename, ctype, timeout=180):
    with open(path, "rb") as fh:
        r = await client.post("/documents", headers=headers, files=[("files", (filename, fh.read(), ctype))])
    assert r.status_code == 201, r.text
    doc_id = r.json()[0]["id"]
    status = "pending"
    for _ in range(timeout):
        status = (await client.get(f"/documents/{doc_id}", headers=headers)).json()["status"]
        if status in ("ready", "failed"):
            break
        await asyncio.sleep(1)
    assert status == "ready", f"ingest ended in {status}"
    return doc_id


async def _ask(client, headers, chat_id, question):
    tokens, done, event = [], None, None
    async with client.stream(
        "POST", f"/chats/{chat_id}/query", headers=headers, json={"question": question}
    ) as resp:
        assert resp.status_code == 200, resp.status_code
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = json.loads(line.split(":", 1)[1].strip())
                if event == "token":
                    tokens.append(payload)
                elif event == "done":
                    done = payload
    return "".join(tokens), done


async def _new_chat(client, headers):
    return (await client.post("/chats", headers=headers, json={"title": "t"})).json()["id"]


async def test_grounded_answer_with_citations(client, auth_headers):
    await _upload_and_wait(
        client, auth_headers, ASSETS / "SAMPLE_PDF.pdf", "SAMPLE_PDF.pdf", "application/pdf"
    )
    chat_id = await _new_chat(client, auth_headers)
    answer, done = await _ask(client, auth_headers, chat_id, "What is this document about?")
    assert answer.strip()
    assert done["grounded"] is True
    assert done["citations"], "expected at least one citation"
    top = done["citations"][0]
    assert top["filename"] == "SAMPLE_PDF.pdf"
    assert "page" in top and "snippet" in top


async def test_absent_information_is_refused(client, auth_headers):
    await _upload_and_wait(client, auth_headers, ASSETS / "SAMPLE_MD.md", "SAMPLE_MD.md", "text/markdown")
    chat_id = await _new_chat(client, auth_headers)
    answer, done = await _ask(
        client,
        auth_headers,
        chat_id,
        "What was the closing price of Tesla stock on the moon in 1823?",
    )
    # Either the score gate rejects, or the prompt makes the model refuse — both are valid.
    assert done["grounded"] is False or "couldn't find" in answer.lower()


async def test_conversation_history_persisted(client, auth_headers):
    await _upload_and_wait(client, auth_headers, ASSETS / "SAMPLE_PDF.pdf", "doc.pdf", "application/pdf")
    chat_id = await _new_chat(client, auth_headers)
    await _ask(client, auth_headers, chat_id, "Summarize this document.")
    msgs = (await client.get(f"/chats/{chat_id}/messages", headers=auth_headers)).json()
    roles = [m["role"] for m in msgs]
    assert roles[:2] == ["user", "assistant"]


async def test_tenant_isolation(client, auth_headers):
    # User A ingests a document.
    await _upload_and_wait(
        client, auth_headers, ASSETS / "SAMPLE_PDF.pdf", "SAMPLE_PDF.pdf", "application/pdf"
    )
    # User B has no documents -> cannot retrieve A's content.
    email = f"user_{uuid.uuid4().hex[:8]}@test.dev"
    pw = "s3cure-password"
    await client.post("/auth/register", json={"email": email, "password": pw})
    tok = (await client.post("/auth/jwt/login", data={"username": email, "password": pw})).json()
    headers_b = {"Authorization": f"Bearer {tok['access_token']}"}
    chat_id = await _new_chat(client, headers_b)
    _, done = await _ask(client, headers_b, chat_id, "What is this document about?")
    assert done["grounded"] is False


async def test_upload_rejects_unsupported_type(client, auth_headers):
    r = await client.post(
        "/documents", headers=auth_headers, files=[("files", ("x.zip", b"data", "application/zip"))]
    )
    assert r.status_code == 415
