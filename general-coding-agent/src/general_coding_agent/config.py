"""Centralized configuration management for general_coding_agent."""

from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    llm_provider: Literal["gemini", "openai", "anthropic", "vllm"] = Field(
        default="gemini",
        validation_alias="LLM_PROVIDER",
        description="Active LLM provider backend",
    )
    llm_model: str = Field(
        default="gemini-3.5-flash-lite",
        validation_alias="LLM_MODEL",
        description="Model name for reasoning/planning. Free tier: 15 RPM / 500 RPD per model.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for deterministic agent reasoning",
    )
    llm_timeout_seconds: float = Field(default=60.0, gt=0, description="Per-attempt provider request timeout")
    # Transient-error retry (429 rate limits, 5xx). The provider SDK is the SINGLE retry owner
    # (LEVEL_0 doc: "SDK is the single transient transport retry owner; do not stack graph/tool
    # retries on top"). Free tier is per-MINUTE limited (15 RPM), so the FIRST backoff must be long
    # enough to cross a 60s window: 10 -> 20 -> 40s (exp base 2, 3 retries) restores the RPM budget.
    # The default 1s initial delay wastes early retries against a minute-long window, which is why
    # 429s were crashing runs. attempts counts the original call (4 = 1 call + 3 retries).
    llm_max_retries: int = Field(
        default=4, ge=1,
        validation_alias="LLM_MAX_RETRIES",
        description="Total model-call attempts (original + retries) on transient errors",
    )
    # These are FREE-TIER defaults (per-minute limited). A higher-tier key with a large RPM has no
    # reason to wait 10s — override all three via env (LLM_MAX_RETRIES / LLM_RETRY_INITIAL_DELAY /
    # LLM_RETRY_EXP_BASE). Nothing here is baked in for "any user"; the window just has to be crossed.
    llm_retry_initial_delay: float = Field(
        default=10.0, gt=0,
        validation_alias="LLM_RETRY_INITIAL_DELAY",
        description="Seconds before the first retry (must approach the RPM window to clear a 429)",
    )
    llm_retry_exp_base: float = Field(
        default=2.0, ge=1.0,
        validation_alias="LLM_RETRY_EXP_BASE",
        description="Multiplier applied to the delay after each retry (10 -> 20 -> 40s)",
    )
    # A daily-quota (RPD) 429 cannot be waited out by retry, but each Gemini model has its OWN quota.
    # So after retries are exhausted with a RATE-LIMIT error only, fall back to the next model here.
    # This is a FREE-TIER convenience (the minimal stand-in for LiteLLM routing, which is deferred);
    # comma-separated, env-overridable, and the active model is skipped if it appears in the list.
    llm_fallback_models: str = Field(
        default="gemini-3.1-flash-lite",
        validation_alias="LLM_FALLBACK_MODELS",
        description="Comma-separated sibling models to try on a rate-limit 429 after retries; '' disables",
    )

    # Provider API keys & endpoints
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    vllm_api_key: str | None = Field(default=None, validation_alias="VLLM_API_KEY")
    vllm_base_url: str | None = Field(default=None, validation_alias="VLLM_BASE_URL")

    # Workspace & Execution Sandboxing
    workspace_root: Path = Field(
        default_factory=lambda: Path.cwd().resolve(),
        validation_alias="WORKSPACE_ROOT",
        description="Root directory for workspace file operations and commands",
    )
    command_timeout_seconds: int = Field(
        default=30,
        gt=0,
        description="Timeout for shell command execution",
    )
    command_output_bytes: int = Field(default=1_000_000, gt=0, description="Combined stdout/stderr byte limit; exceeding it terminates the command")
    max_supervisor_iterations: int = Field(
        default=5,
        ge=1,
        validation_alias="MAX_SUPERVISOR_ITERATIONS",
        description="Maximum iterations the supervisor can loop with worker agents before forcing completion",
    )
    graph_step_limit: int = Field(default=500, gt=0, description="Native graph recursion ceiling; shared call/time budgets also apply")
    model_call_limit: int = Field(default=100, gt=0, description="Logical model calls per run, including summaries and memory capture")
    tool_call_limit: int = Field(default=200, gt=0, description="Tool attempts per run; preserved across approval resumes")
    run_timeout_seconds: float = Field(default=900.0, gt=0, description="Active execution time per run, excluding time waiting for a human")
    memory_capture_mode: Literal["explicit", "review"] = Field(default="explicit", description="Save source-quoted explicit statements automatically, or retain proposals for review")
    # fs_read pages large files instead of dumping them: at most `read_page_lines` lines per call
    # (agent pages with offset), and any single line is capped at `max_line_chars`.
    read_page_lines: int = Field(default=400, gt=0, description="Max lines fs_read returns per call")
    file_max_bytes: int = Field(default=2_000_000, gt=0, description="Maximum UTF-8 text file bytes for filesystem tools")
    max_line_chars: int = Field(default=1000, gt=0, description="Per-line char cap in fs_read")
    # Artifact offloading: any tool result larger than the threshold is stored as an
    # artifact and only a preview + reference reaches the model (it can read_artifact the rest).
    artifact_threshold_chars: int = Field(
        default=2000,
        gt=0,
        validation_alias="ARTIFACT_THRESHOLD_CHARS",
        description="Tool-result size (chars) above which the result is offloaded to an artifact",
    )
    artifact_preview_chars: int = Field(
        default=600,
        gt=0,
        description="How many leading chars of an offloaded result to show the model as a preview",
    )
    artifact_read_chars: int = Field(default=4000, gt=0, description="Maximum artifact characters per read")
    artifacts_dir: Path = Field(
        default=Path(".agent_artifacts"),
        description="Local artifact store dir (stands in for Ceph/S3 object storage)",
    )

    # Memory & Compaction
    memory_db_path: Path = Field(
        default=Path(".agent_memory.db"),
        description="Path to local SQLite checkpointer DB (thread conversation state)",
    )
    store_db_path: Path = Field(
        default=Path(".agent_store.db"),
        description="Path to local SQLite long-term memory store (cross-thread facts)",
    )
    input_max_chars: int = Field(default=100_000, gt=0, validation_alias="INPUT_MAX_CHARS", description="Maximum characters allowed in a user text request or resume input")
    compaction_token_threshold: int = Field(
        default=2000,
        gt=200,
        validation_alias="COMPACTION_TOKEN_THRESHOLD",
        description="Approx. token count of working messages that triggers compaction",
    )
    model_context_tokens: int = Field(default=16000, gt=0, description="Application context allowance; token estimates include output reserve and tool schemas")
    model_output_reserve_tokens: int = Field(default=4000, gt=0, description="Provider output cap, also reserved in context admission")
    context_safety_tokens: int = Field(default=1_000, ge=0, validation_alias="CONTEXT_SAFETY_TOKENS", description="Conservative margin reserved for token estimation errors and tool overhead")
    summary_max_tokens: int = Field(default=2000, gt=0, description="Maximum estimated tokens in a compacted segment")

    @model_validator(mode="after")
    def validate_limits(self) -> "Settings":
        usable = self.model_context_tokens - self.model_output_reserve_tokens - self.context_safety_tokens
        if usable <= 0:
            raise ValueError("model context must exceed output and safety reserves")
        if self.summary_max_tokens >= usable:
            raise ValueError("summary_max_tokens must fit usable model input")
        if self.compaction_token_threshold >= usable:
            raise ValueError("compaction threshold must be below usable model input")
        if self.artifact_preview_chars > self.artifact_threshold_chars:
            raise ValueError("artifact preview cannot exceed offload threshold")
        return self
    memory_context_chars: int = Field(default=8000, gt=0, description="Total durable-memory prompt character cap")
    memory_content_chars: int = Field(default=2000, gt=0, description="Maximum characters in one durable fact")
    memory_scan_records: int = Field(default=1000, gt=0, description="Maximum candidate records inspected per memory scope")
    memory_recall_records: int = Field(default=10, gt=0, description="Maximum records included in a memory response")
    memory_max_records: int = Field(default=10000, gt=0, description="Maximum retained facts per owner/scope, including retired facts")
    memory_max_revisions: int = Field(default=100000, gt=0, description="Maximum retained fact revisions per owner/scope; no automatic destructive purge")

    # Identity (namespaces long-term memory; server-derived in a multi-user deployment)
    user_id: str = Field(default="local_user", validation_alias="USER_ID")
    project_id: str = Field(default="default_project", validation_alias="PROJECT_ID")

    # Skills
    skills_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "skills",
        validation_alias="SKILLS_DIR",
        description="Path to skills directory containing SKILL.md packages",
    )

    def resolve_workspace_path(self, path: str | Path) -> Path:
        """Resolve a path and guarantee it stays inside workspace_root.

        Uses Path.is_relative_to (not string prefix) so a sibling like
        '<root>-evil' cannot masquerade as being inside '<root>'.
        """
        raw = Path(path)
        root = self.workspace_root.resolve()
        resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        if not resolved.is_relative_to(root):
            raise PermissionError(f"Access denied: '{resolved}' escapes workspace root '{root}'")
        return resolved


settings = Settings()
