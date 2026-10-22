"""Real socket clients and isolated browser sessions; no provider mocks."""

import json
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request

import pytest
from playwright.sync_api import sync_playwright


def pytest_addoption(parser):
    parser.addoption("--live-book", action="store_true", help="Run the complete real-provider browser book workflow")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live-book"):
        for item in items:
            if "live_book" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Pass --live-book to use real model/search quota"))


class Studio:
    def __init__(self):
        self.url = os.environ.get("BOOK_STUDIO_TEST_URL", "http://127.0.0.1:8000").rstrip("/")
        self.registration_key = os.environ.get("REGISTRATION_KEY")
        if not self.registration_key:
            pytest.fail("REGISTRATION_KEY is required; point BOOK_STUDIO_TEST_URL at a running test deployment")

    def call(self, path, *, method="GET", payload=None, form=None, token=None, headers=None, expected=200):
        headers = dict(headers or {})
        data = None
        if token:
            headers["Authorization"] = "Bearer " + token
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode()
        if form is not None:
            data = urllib.parse.urlencode(form).encode()
        req = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                code, body, response_headers = response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            code, body, response_headers = error.code, error.read(), error.headers
        assert code == expected, f"{method} {path} returned {code}, expected {expected}"
        return body, response_headers

    def json(self, path, **kwargs):
        return json.loads(self.call(path, **kwargs)[0])

    def credentials(self):
        return {"username": "check_" + secrets.token_hex(5), "password": secrets.token_urlsafe(24)}

    def register(self, credentials):
        return self.json("/auth/register", method="POST", payload=credentials,
                         headers={"X-Registration-Key": self.registration_key}, expected=201)

    def token(self, credentials):
        return self.json("/auth/token", method="POST", form=credentials)["access_token"]


@pytest.fixture
def studio():
    client = Studio()
    assert client.json("/health")["status"] == "ok"
    return client


@pytest.fixture
def accounts(studio):
    users = [studio.credentials(), studio.credentials()]
    for user in users:
        studio.register(user)
        user["token"] = studio.token(user)
    return users


@pytest.fixture
def page():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        yield page
        context.close()
        browser.close()
        assert not errors, "Uncaught browser errors: " + "; ".join(errors)
