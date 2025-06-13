"""Utilities for sanitizing LLM outputs."""

from __future__ import annotations

import re
from typing import Iterable, List


class OutputFilter:
    """Filter text based on blocked patterns and PII masking."""

    def __init__(self, blocked_patterns: Iterable[str] | None = None, mask_pii: bool = True) -> None:
        self.patterns: List[re.Pattern[str]] = [re.compile(p) for p in blocked_patterns or []]
        self.mask_pii = mask_pii

        self.api_key_re = re.compile(r"[A-Za-z0-9]{32,}")
        self.email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        self.phone_re = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
        self.ssn_re = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        self.credit_card_re = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")

    def filter(self, text: str) -> str:
        """Validate and optionally sanitize ``text``."""
        for pattern in self.patterns:
            if pattern.search(text):
                raise ValueError("Response contains blocked content")

        if self.mask_pii:
            text = self.api_key_re.sub("[REDACTED]", text)
            text = self.email_re.sub("[EMAIL]", text)
            text = self.phone_re.sub("[PHONE]", text)
            text = self.ssn_re.sub("[SSN]", text)
            text = self.credit_card_re.sub("[CARD]", text)

        return text
