import types

from app.chat_settings import ChatSettings
from app.services.chat_service import split_window


def _msg(content: str, i: int):
    return types.SimpleNamespace(content=content, created_at=i, role="user")


def _settings(**kw) -> ChatSettings:
    base = dict(
        top_k=5,
        min_score=0.3,
        history_turns=2,
        context_token_budget=1000,
        temperature=0.0,
        document_ids=None,
    )
    base.update(kw)
    return ChatSettings(**base)


def test_small_history_kept_verbatim():
    msgs = [_msg("hi there", i) for i in range(3)]
    window, to_summarize = split_window(msgs, _settings(history_turns=2, context_token_budget=8000))
    assert to_summarize == []
    assert len(window) == 3


def test_older_turns_summarized_past_budget():
    # ~1000 tokens each; budget 1000 -> only the history_turns floor stays verbatim.
    msgs = [_msg("w" * 4000, i) for i in range(6)]
    window, to_summarize = split_window(msgs, _settings(history_turns=2, context_token_budget=1000))
    assert len(window) == 2
    assert len(to_summarize) == 4
    # boundary is correct: everything to summarize is older than the window
    assert to_summarize[-1].created_at < window[0].created_at


def test_history_turns_floor_wins_over_budget():
    msgs = [_msg("z" * 8000, i) for i in range(4)]
    window, _ = split_window(msgs, _settings(history_turns=3, context_token_budget=1000))
    assert len(window) == 3  # floor kept even though each exceeds the budget
