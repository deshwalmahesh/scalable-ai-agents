"""Chat model factory.

One place to swap providers. Gemini is the default and uses its NATIVE LangChain client
(ChatGoogleGenerativeAI) rather than the OpenAI-compat endpoint: the native client
round-trips Gemini 3's thought-signatures, which multi-step tool calling requires
(the compat endpoint drops them and 400s). openai/anthropic are drop-in alternatives.
"""

from typing import Any
from langchain_core.runnables import Runnable
from general_coding_agent.config import settings


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> Runnable:
    """Instantiate a tool-calling chat model for the active provider.

    The provider SDK owns transient-error retry (see config). For Gemini the default backoff
    (1s initial) is too short for the free tier's per-minute rate limit, so we bind a tuned
    HttpRetryOptions (10->20->40s) onto every call. The returned object still supports
    bind_tools()/with_structured_output() and works with create_react_agent.
    """
    provider = (provider or settings.llm_provider).lower()
    model = model or settings.llm_model
    temperature = settings.llm_temperature if temperature is None else temperature
    kwargs.setdefault("max_retries", settings.llm_max_retries)
    output_limit = kwargs.pop("max_tokens", settings.model_output_reserve_tokens)
    if type(output_limit) is not int or not 0 < output_limit <= settings.model_output_reserve_tokens:
        raise ValueError("max_tokens must fit the configured output reserve")
    if any(key in kwargs for key in ("max_output_tokens", "max_completion_tokens", "max_tokens_to_sample")):
        raise ValueError("Use max_tokens for the provider-independent output limit")
    kwargs["max_tokens"] = output_limit
    kwargs.setdefault("timeout", settings.llm_timeout_seconds)

    if provider in ("gemini", "google"):
        from google.genai.types import HttpOptions, HttpRetryOptions
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_google_genai.chat_models import GoogleRateLimitError

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")

        # max_retries alone only sets attempt COUNT (default delays 1,2,4s — too short to clear a
        # per-minute 429). Bind the full retry policy so the first wait actually crosses the window.
        retry = HttpRetryOptions(
            attempts=settings.llm_max_retries,
            initial_delay=settings.llm_retry_initial_delay,
            exp_base=settings.llm_retry_exp_base,
        )

        def _bound(m: str):
            chat = ChatGoogleGenerativeAI(
                model=m, temperature=temperature, google_api_key=settings.gemini_api_key, **kwargs,
            )
            return chat.bind(http_options=HttpOptions(retry_options=retry))

        primary = _bound(model)
        # After retries are exhausted with a RATE-LIMIT error, try a sibling model (separate quota).
        # Scoped to GoogleRateLimitError so real errors (e.g. a 400 bad request) still surface loudly.
        fallbacks = [_bound(m) for m in
                     (x.strip() for x in settings.llm_fallback_models.split(",")) if m and m != model]
        if fallbacks:
            return primary.with_fallbacks(fallbacks, exceptions_to_handle=(GoogleRateLimitError,))
        return primary

    if provider in ("openai", "vllm"):
        from langchain_openai import ChatOpenAI

        api_key = settings.vllm_api_key or settings.openai_api_key
        base_url = settings.vllm_base_url or settings.openai_base_url
        if not api_key:
            raise ValueError(f"API key for {provider} is not set. Check your .env file.")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            streaming=True,
            **kwargs,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Check your .env file.")
        return ChatAnthropic(model=model, temperature=temperature, api_key=settings.anthropic_api_key, streaming=True, **kwargs)

    raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported: gemini, openai, anthropic, vllm")

