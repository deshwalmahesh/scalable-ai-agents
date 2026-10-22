"""Real browser authentication, rendering, and provider-backed book workflow."""

import re
import time

import pytest
from playwright.sync_api import expect

from book_studio.writer import format_issues


def sign_in(page, studio, credentials):
    page.goto(studio.url)
    page.get_by_label("Username", exact=True).fill(credentials["username"])
    page.get_by_label("Password", exact=True).fill(credentials["password"])
    page.locator("#auth-submit").click()
    expect(page.locator("#workspace")).to_be_visible()


def test_browser_signup_keyboard_session_and_mobile(page, studio, tmp_path):
    credentials = studio.credentials()
    page.goto(studio.url)
    page.screenshot(path=tmp_path / "sign-in-desktop.png", full_page=True)
    page.get_by_role("button", name="Create account", exact=True).click()
    page.get_by_label("Username", exact=True).fill(credentials["username"])
    page.get_by_label("Password", exact=True).fill(credentials["password"])
    page.get_by_label("Registration key", exact=True).fill("wrong-key")
    page.locator("#auth-submit").click()
    expect(page.locator("#auth-error")).to_contain_text("Registration is not allowed")
    page.get_by_label("Registration key", exact=True).fill(studio.registration_key)
    page.get_by_label("Registration key", exact=True).press("Enter")
    expect(page.locator("#workspace")).to_be_visible()
    expect(page.locator("#empty-desk")).to_be_visible()
    assert page.evaluate("() => localStorage.length") == 0
    page.reload()
    expect(page.locator("#account-name")).to_have_text(credentials["username"])
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path=tmp_path / "empty-mobile.png", full_page=True)
    page.context.set_offline(True)
    page.get_by_role("button", name="Refresh", exact=True).click()
    expect(page.locator("#workspace-error")).to_contain_text("Cannot reach the studio")
    page.context.set_offline(False)
    page.get_by_role("button", name="Refresh", exact=True).click()
    expect(page.locator("#workspace-error")).to_be_hidden()
    page.get_by_role("button", name="Sign out", exact=True).click()
    expect(page.get_by_label("Username", exact=True)).to_be_focused()
    assert page.evaluate("() => sessionStorage.getItem('book-token')") is None
    expect(page.locator("#manuscript")).to_be_empty()
    page.get_by_label("Username", exact=True).fill(credentials["username"])
    page.get_by_label("Password", exact=True).fill(credentials["password"])
    page.get_by_label("Password", exact=True).press("Enter")
    expect(page.locator("#workspace")).to_be_visible()
    page.evaluate("() => sessionStorage.setItem('book-token', 'expired-token')")
    page.reload()
    expect(page.locator("#auth-error")).to_contain_text("session has expired")
    expect(page.locator("#workspace")).to_be_hidden()


def test_renderer_cannot_execute_retrieved_content(page, studio):
    page.goto(studio.url)
    result = page.evaluate("""(markdown) => {
      const host = document.createElement('article');
      host.append(renderMarkdown(markdown));
      document.body.append(host);
      const result = { text: host.textContent, scripts: host.querySelectorAll('script,img').length, links: [...host.querySelectorAll('a')].map(a => a.href), stolen: window.stolen || 0 };
      host.remove(); return result;
    }""", '## Chapter\n\n<img src=x onerror="window.stolen=1">\n\n<script>window.stolen=2</script>\n\n[1] <javascript:alert(1)> <https://?> <https://example.com/source>')
    assert result["scripts"] == 0 and result["stolen"] == 0
    assert "<img" in result["text"]
    assert result["links"] == ["https://example.com/source"]


@pytest.mark.live_book
@pytest.mark.parametrize("brief", [None, """Title: The Moon Above Us: A Beginner's Guide
Audience: Curious adult readers with no astronomy background.
Explain the Moon, why its appearance changes through a month, and how a beginner can observe it. Use welcoming plain English, explain terms, and use authoritative primary sources such as NASA. Avoid unrelated planetary histories, equipment shopping and unsupported predictions. Write exactly three chapters of 600–900 prose words with numbered citations and one final Takeaway per chapter.
"""], ids=["upi-default", "custom-topic"])
def test_full_browser_book_with_pause_feedback_and_downloads(page, studio, accounts, tmp_path, brief):
    owner, other = accounts
    sign_in(page, studio, owner)
    if brief is None:
        page.get_by_role("button", name="+ New default book", exact=True).click()
    else:
        page.locator("#custom-book summary").click()
        page.get_by_label("Book brief", exact=True).fill(brief)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=tmp_path / "custom-brief-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.get_by_role("button", name="Start custom book", exact=True).click()
    expect(page.locator("#book-desk")).to_be_visible()
    page.get_by_role("button", name="Pause book", exact=True).click()
    expect(page.locator("#job-status")).to_have_text("Paused", timeout=180000)
    job_id = page.locator('.job-button[aria-current="true"]').get_attribute("data-job")
    endpoint = "/jobs/" + job_id
    created = studio.json(endpoint, token=owner["token"])
    if brief is not None:
        assert created["brief"] == brief.strip()
        expect(page.locator("#book-title")).to_have_text("The Moon Above Us: A Beginner's Guide")
    else:
        assert created["brief"].startswith("Title: Pay Me on UPI:")
    page.reload()
    expect(page.locator("#job-status")).to_have_text("Paused")
    page.get_by_role("button", name="Resume book", exact=True).click()
    expect(page.locator("#notice")).to_contain_text("resumed")
    deadline = time.monotonic() + 3300
    revised = False
    original_opening = None
    approvals = 0
    while time.monotonic() < deadline:
        job = studio.json(endpoint, token=owner["token"])
        assert job["status"] != "failed", job["error"]
        if job["status"] == "completed":
            break
        if job["status"] == "awaiting_review":
            expect(page.locator("#review-controls")).to_be_visible(timeout=15000)
            expect(page.locator("#manuscript")).to_contain_text("Takeaway:")
            body = job["review_preview"].split("\n\n", 1)[1].split("\n### References", 1)[0].strip()
            assert not format_issues(body)
            if brief is not None:
                assert "UPI" not in body and "NPCI" not in body
            if not revised:
                page.get_by_label("Revision feedback", exact=True).fill("   ")
                page.get_by_role("button", name="Request changes", exact=True).click()
                expect(page.locator("#feedback-error")).to_contain_text("Describe what")
                opening = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)[0]
                original_opening = opening
                audience = "a first-time shop owner" if brief is None else "a beginner astronomy reader"
                feedback = f"Replace this opening with a distinct, plain-English opening for {audience}: {opening}. Do not reuse this exact sentence as the opening. Preserve all verified claims, citations and required chapter structure."
                page.get_by_label("Revision feedback", exact=True).fill(feedback[:2000])
                page.wait_for_timeout(4500)
                expect(page.get_by_label("Revision feedback", exact=True)).to_have_value(feedback[:2000])
                with page.expect_response(lambda r: r.url.endswith(endpoint + "/review") and r.request.method == "POST") as sent:
                    page.get_by_role("button", name="Request changes", exact=True).click()
                assert sent.value.status == 202
                expect(page.locator("#notice")).to_contain_text("Changes requested")
                revised = True
            else:
                if approvals == 0:
                    assert re.split(r"(?<=[.!?])\s+", body, maxsplit=1)[0] != original_opening
                page.screenshot(path=tmp_path / f"review-{approvals + 1}.png", full_page=True)
                with page.expect_response(lambda r: r.url.endswith(endpoint + "/review") and r.request.method == "POST") as sent:
                    page.get_by_role("button", name="Approve chapter", exact=True).click()
                assert sent.value.status == 202
                approvals += 1
            page.wait_for_timeout(5000)
        else:
            page.wait_for_timeout(2000)
    else:
        raise AssertionError("Real book did not complete within 55 minutes")
    assert revised and approvals == 3
    expect(page.locator("#job-status")).to_have_text("Complete", timeout=15000)
    for n in range(1, 4):
        page.locator(f'button[data-view="{n}"]').click()
        expect(page.locator("#manuscript")).to_contain_text(f"Chapter {n}:")
        with page.expect_download() as event:
            page.get_by_role("button", name="Download chapter", exact=True).click()
        event.value.save_as(tmp_path / f"chapter-{n}.md")
    for filename, label in (("book.pdf", "Download PDF"), ("book.md", "Download Markdown")):
        with page.expect_download() as event:
            page.get_by_role("button", name=label, exact=True).click()
        event.value.save_as(tmp_path / filename)
    assert (tmp_path / "book.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path / "book.md").read_text().count("## Chapter ") == 3
    studio.call(endpoint + "/artifacts/book.pdf", token=other["token"], expected=404)
    # Hold a real authenticated response until after sign-out; no fabricated API data.
    held = []
    downloads = []
    page.on("download", lambda event: downloads.append(event.suggested_filename))
    page.route("**/artifacts/book.pdf", lambda route: held.append((route, route.fetch())))
    page.get_by_role("button", name="Download PDF", exact=True).click()
    deadline = time.monotonic() + 25
    while not held and time.monotonic() < deadline:
        page.wait_for_timeout(100)
    assert held, "Download request did not reach the real service"
    page.get_by_role("button", name="Sign out", exact=True).click()
    held[0][0].fulfill(response=held[0][1])
    page.wait_for_timeout(1000)
    assert not downloads, "An old session downloaded private content after sign-out"
    page.unroute("**/artifacts/book.pdf")
    sign_in(page, studio, owner)
    expect(page.locator("#job-status")).to_have_text("Complete")
    # Multiple real jobs exercise the sidebar's intrinsic width, even when paused.
    for _ in range(2):
        page.get_by_role("button", name="+ New default book", exact=True).click()
        expect(page.get_by_role("button", name="Pause book", exact=True)).to_be_visible()
        page.get_by_role("button", name="Pause book", exact=True).click()
        expect(page.locator("#job-status")).to_have_text("Paused", timeout=180000)
    held.clear()
    page.route("**/jobs/" + job_id, lambda route: held.append((route, route.fetch())))
    page.locator(f'button[data-job="{job_id}"]').click()
    deadline = time.monotonic() + 25
    while not held and time.monotonic() < deadline:
        page.wait_for_timeout(100)
    assert held, "Selected book request did not reach the real service"
    expect(page.locator("#book-desk")).to_be_hidden()
    expect(page.locator("#notice")).to_contain_text("Loading book")
    for route, response in held:
        route.fulfill(response=response)
    page.unroute("**/jobs/" + job_id)
    expect(page.locator("#job-status")).to_have_text("Complete")
    page.screenshot(path=tmp_path / "completed-desktop.png", full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path=tmp_path / "completed-mobile.png", full_page=True)
