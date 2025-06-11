import pytest

from core.model_policy import ModelSelector


def test_selector_prefers_policy_model():
    metadata = [{"id": "a"}, {"id": "b"}]
    selector = ModelSelector(metadata, {"assistant": "a"})
    assert selector.model_for_role("assistant") == "a"


def test_selector_falls_back_to_default():
    metadata = [{"id": "a"}, {"id": "b"}]
    selector = ModelSelector(metadata, {"critic": "c", "default": "b"})
    assert selector.model_for_role("critic") == "b"


def test_selector_first_available_when_missing():
    metadata = [{"id": "a"}]
    selector = ModelSelector(metadata, {"critic": "c"})
    assert selector.model_for_role("critic") == "a"


def test_selector_raises_for_no_models():
    with pytest.raises(ValueError):
        ModelSelector([], {"assistant": "a"})
