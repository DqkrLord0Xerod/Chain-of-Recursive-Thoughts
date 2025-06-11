import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location(
    "cache_manager", os.path.join(ROOT, "core", "cache_manager.py")
)
cache_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_manager)
CacheManager = cache_manager.CacheManager


def test_memory_lru():
    cm = CacheManager(enabled=True, memory_size=2)
    cm.set(("a", "1"), "x")
    cm.set(("b", "1"), "y")
    cm.set(("c", "1"), "z")
    assert cm.get(("a", "1")) is None
    assert cm.get(("b", "1")) == "y"


def test_disk_persistence(tmp_path):
    path = tmp_path / "cache.pkl"
    cm1 = CacheManager(enabled=True, memory_size=1, disk_path=str(path), disk_size=2)
    cm1.set(("k", "1"), "v")

    cm2 = CacheManager(enabled=True, memory_size=1, disk_path=str(path), disk_size=2)
    assert cm2.get(("k", "1")) == "v"

