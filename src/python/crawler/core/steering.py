"""Layer 1 — topic steering (pure).

A decaying keyword bag that scores text for relevance. This feeds the relevance
signal; steering biases the walk, it does not (by itself) gate it.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(value.lower())


@dataclass
class Keyword:
    term: str
    weight: float
    updated_at: float


class TopicState:
    # Bounded: the control API can add keywords forever; an endless run must not
    # accumulate them. Weakest (decayed) terms are evicted past the cap.
    MAX_KEYWORDS = 256

    def __init__(self) -> None:
        self._keywords: dict[str, Keyword] = {}

    def _decayed_weight(self, keyword: Keyword, now: float) -> float:
        age_minutes = max((now - keyword.updated_at) / 60.0, 0.0)
        return keyword.weight * (0.96 ** age_minutes)

    def _evict_weakest(self, now: float) -> None:
        while len(self._keywords) > self.MAX_KEYWORDS:
            weakest = min(
                self._keywords.values(), key=lambda keyword: self._decayed_weight(keyword, now)
            )
            del self._keywords[weakest.term]

    def set_keywords(self, keywords: Iterable[str]) -> list[str]:
        now = time.time()
        normalized: dict[str, Keyword] = {}
        for raw in keywords:
            for token in tokenize(str(raw)):
                if token:
                    normalized[token] = Keyword(token, 1.0, now)
        self._keywords = normalized
        self._evict_weakest(now)
        return self.keywords

    def add_keywords(self, keywords: Iterable[str]) -> list[str]:
        now = time.time()
        for raw in keywords:
            for token in tokenize(str(raw)):
                if token:
                    current = self._keywords.get(token)
                    weight = min((current.weight if current else 0.0) + 1.0, 5.0)
                    self._keywords[token] = Keyword(token, weight, now)
        self._evict_weakest(now)
        return self.keywords

    @property
    def keywords(self) -> list[str]:
        return sorted(self._keywords)

    def score_text(self, *values: str) -> float:
        if not self._keywords:
            return 0.1

        tokens = tokenize(" ".join(value for value in values if value))
        if not tokens:
            return 0.0

        now = time.time()
        score = 0.0
        token_set = set(tokens)
        for keyword in self._keywords.values():
            age_minutes = max((now - keyword.updated_at) / 60.0, 0.0)
            decayed_weight = keyword.weight * (0.96 ** age_minutes)
            if keyword.term in token_set:
                score += decayed_weight
            else:
                score += sum(
                    0.35 * decayed_weight
                    for token in token_set
                    if keyword.term in token or token in keyword.term
                )
        return score
