import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402
from core.model_policy import ModelSelector  # noqa: E402


def test_map_roles_with_default():
    metadata = [{"id": "a"}, {"id": "b"}]
    selector = ModelSelector(metadata, {"assistant": "b", "default": "a"})
    result = selector.map_roles(["assistant", "critic"])
    assert result == {"assistant": "b", "critic": "a"}
