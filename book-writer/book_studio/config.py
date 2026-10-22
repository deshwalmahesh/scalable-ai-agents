"""Shared environment loading and configuration validation."""

from __future__ import annotations

import ipaddress
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from string import Formatter
from urllib.parse import urlparse

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_ENV_FILE = Path(".env")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
PROMPTS_FILE = Path(__file__).with_name("prompts.json")
DEFAULTS = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
DEFAULT_BRIEF = DEFAULTS["default_brief"]
DOMAIN_PATTERN = r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,63}"


class PromptPair(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    system: str = Field(min_length=1, max_length=20000)
    human: str = Field(min_length=1, max_length=20000)


class ResearchPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    domains: list[str] = Field(max_length=32)
    seeds: list[list[list[str]]] = Field(min_length=3, max_length=3)
    excluded_titles: list[str] = Field(min_length=3, max_length=3)
    fallback_queries: list[str] = Field(min_length=2, max_length=2)


def source_allowed(url: str, domains: list[str] | tuple[str, ...]) -> bool:
    """Accept public HTTPS source URLs, optionally restricted to trusted domains."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
            return False
        try:
            if not ipaddress.ip_address(host).is_global:
                return False
        except ValueError:
            if not re.fullmatch(DOMAIN_PATTERN, host) or host.endswith((".localhost", ".local", ".internal")):
                return False
        return not domains or any(host == domain or host.endswith("." + domain) for domain in domains)
    except ValueError:
        return False


def template_fields(template: str) -> set[str]:
    fields = set()
    for _, field, spec, conversion in Formatter().parse(template):
        if field is not None:
            if not field.isidentifier() or spec or conversion:
                raise ValueError("Prompt placeholders must be plain named fields")
            fields.add(field)
    return fields


def _merge_config(base: dict, override: dict) -> dict:
    if not isinstance(override, dict) or override.keys() - base.keys():
        raise ValueError("Prompt configuration contains unknown fields")
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge_config(base[key], value) if isinstance(base[key], dict) else value
    return result


def validate_profile(profile: dict) -> dict:
    if set(profile) != {"prompts", "research"} or set(profile["prompts"]) != set(DEFAULTS["profiles"]["upi"]["prompts"]):
        raise ValueError("A book profile must contain every agent prompt and research policy")
    for name, values in profile["prompts"].items():
        pair = PromptPair.model_validate(values)
        for role, template in pair.model_dump().items():
            expected = template_fields(DEFAULTS["profiles"]["upi"]["prompts"][name][role])
            actual = template_fields(template)
            if not template.strip() or expected - actual or actual - (expected | {"brief"}):
                raise ValueError(f"Invalid {name}.{role} placeholders: preserve {sorted(expected)}; brief is also available")
    policy = ResearchPolicy.model_validate(profile["research"])
    for domain in policy.domains:
        if not re.fullmatch(DOMAIN_PATTERN, domain):
            raise ValueError("Research domains must be lowercase public hostnames without URL paths")
    for seeds in policy.seeds:
        if len(seeds) > 8:
            raise ValueError("At most eight seed pages per chapter")
        for seed in seeds:
            if len(seed) != 2 or not seed[1].strip() or not source_allowed(seed[0], policy.domains):
                raise ValueError("Research seeds need a permitted HTTPS URL and title")
    for pattern in policy.excluded_titles:
        re.compile(pattern)
    for query in policy.fallback_queries:
        if not query.strip() or template_fields(query) - {"title", "focus"}:
            raise ValueError("Fallback queries may use only title and focus placeholders")
    return deepcopy(profile)


def load_profiles(config: dict[str, str]) -> dict:
    profiles = deepcopy(DEFAULTS)
    # General prompts inherit only shared contracts; topic-specific defaults override them.
    profiles["profiles"]["general"] = _merge_config(
        profiles["profiles"]["upi"], profiles["profiles"]["general"]
    )
    if config.get("BOOK_PROMPTS_FILE"):
        override = json.loads(Path(config["BOOK_PROMPTS_FILE"]).read_text(encoding="utf-8"))
        profiles = _merge_config(profiles, override)
    brief = profiles["default_brief"]
    if not isinstance(brief, str) or not 1 <= len(brief.strip()) <= 20000:
        raise ValueError("The default book brief must contain 1–20000 characters")
    profiles["profiles"] = {name: validate_profile(profile) for name, profile in profiles["profiles"].items()}
    return profiles


def load_config(env_file: Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    values = {key: value for key, value in dotenv_values(env_file).items() if value}
    values.update(os.environ)
    required = ("VLLM_BASE_URL", "VLLM_API_KEY", "TAVILY_API_KEY")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f"Missing configuration: {', '.join(missing)} (checked {env_file} and environment)")
    return values


def load_service_config() -> dict[str, str]:
    config = load_config(Path(os.environ.get("BOOK_ENV_FILE", str(DEFAULT_ENV_FILE))))
    if any(len(config.get(key, "")) < 32 for key in ("JWT_SECRET", "REGISTRATION_KEY")):
        raise RuntimeError("JWT_SECRET and REGISTRATION_KEY must each contain at least 32 characters")
    return config
