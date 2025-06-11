from __future__ import annotations

import os
import pickle
from collections import OrderedDict
from typing import Any, Tuple


class CacheManager:
    """Simple in-memory and optional disk cache manager."""

    def __init__(
        self,
        enabled: bool = True,
        memory_size: int = 128,
        disk_path: str | None = None,
        disk_size: int = 256,
    ) -> None:
        self.enabled = enabled
        self.memory_size = memory_size
        self.disk_path = disk_path
        self.disk_size = disk_size
        self.memory: OrderedDict[Tuple[str, str], Any] = OrderedDict()
        self.disk: OrderedDict[Tuple[str, str], Any] = OrderedDict()
        if self.disk_path:
            self._load_disk()

    # basic LRU management
    def get(self, key: Tuple[str, str]) -> Any | None:
        if not self.enabled:
            return None
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        if key in self.disk:
            value = self.disk[key]
            self.memory[key] = value
            self.memory.move_to_end(key)
            if len(self.memory) > self.memory_size:
                self.memory.popitem(last=False)
            return value
        return None

    def set(self, key: Tuple[str, str], value: Any) -> None:
        if not self.enabled:
            return
        self.memory[key] = value
        self.memory.move_to_end(key)
        if len(self.memory) > self.memory_size:
            self.memory.popitem(last=False)
        if self.disk_path:
            self.disk[key] = value
            if len(self.disk) > self.disk_size:
                self.disk.popitem(last=False)
            self._save_disk()

    def _load_disk(self) -> None:
        if not self.disk_path or not os.path.exists(self.disk_path):
            return
        try:
            with open(self.disk_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.disk = OrderedDict(data)
                while len(self.disk) > self.disk_size:
                    self.disk.popitem(last=False)
        except Exception:
            self.disk = OrderedDict()

    def _save_disk(self) -> None:
        if not self.disk_path:
            return
        try:
            with open(self.disk_path, "wb") as f:
                pickle.dump(self.disk, f)
        except Exception:
            pass

