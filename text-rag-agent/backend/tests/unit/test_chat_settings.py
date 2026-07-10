import pytest
from pydantic import ValidationError

from app.chat_settings import ChatSettingsPatch, resolve_settings


def test_defaults_come_from_config():
    s = resolve_settings(None)
    assert s.top_k >= 1
    assert 0.0 <= s.min_score <= 1.0
    assert s.context_token_budget >= 1000


def test_stored_overrides_win():
    s = resolve_settings({"top_k": 9, "temperature": 0.7})
    assert s.top_k == 9
    assert s.temperature == 0.7


def test_none_values_ignored():
    s = resolve_settings({"top_k": None})
    assert s.top_k == resolve_settings(None).top_k


def test_patch_rejects_out_of_range():
    with pytest.raises(ValidationError):
        ChatSettingsPatch(top_k=999)
    with pytest.raises(ValidationError):
        ChatSettingsPatch(min_score=2.0)
