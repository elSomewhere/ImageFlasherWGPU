"""Layer 1 — multi-signal scoring (pure).

The score that drives frontier priority is a blend of orthogonal signals rather than
topic relevance alone:

    score(link) = inherited + focus·relevance + (1 - focus)·(host_freshness + noise)

- ``relevance``      — keyword/topic match (steerability). Biases the walk; never gates it.
- ``host_freshness`` — favors hosts we've sampled less (anti-tunneling, diversity).
- ``noise``          — the "randomness of what it finds".
- ``focus`` ∈ [0,1]  — one knob: 1 = chase relevance, 0 = pure exploration.

Novelty (unlike-recent-content) is an optional image signal handled separately in
``signals/novelty.py`` because it needs the decoded bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from publicsuffix2 import get_sld


def host_of(url: str) -> str:
    return urlparse(url).hostname or ""


def registrable_domain(url: str) -> str:
    host = host_of(url)
    return get_sld(host, strict=True) or host


@dataclass
class ScoreWeights:
    host_freshness: float = 0.5
    noise: float = 0.25
    inherited: float = 1.0
    image_page: float = 0.2


class ScorePolicy:
    """Pure scorer. Depends only on a TopicState, an RNG, and a host->visits map."""

    def __init__(self, topic_state, rng, host_visits: dict, *, focus: float = 0.5, exploration: float | None = None, weights: ScoreWeights | None = None) -> None:
        self.topic_state = topic_state
        self.rng = rng
        self.host_visits = host_visits
        self.focus = focus
        self.exploration = 1.0 - focus if exploration is None else exploration
        self.weights = weights or ScoreWeights()

    def relevance(self, *texts: str) -> float:
        return self.topic_state.score_text(*texts)

    def host_freshness(self, url: str) -> float:
        return 1.0 / (1.0 + self.host_visits.get(registrable_domain(url), 0))

    def noise(self) -> float:
        return self.rng.random()

    def score_link(self, item) -> float:
        weights = self.weights
        relevance = self.relevance(item.url, item.context)
        exploration = weights.host_freshness * self.host_freshness(item.url) + weights.noise * self.noise()
        exploration_amount = max(0.0, min(1.0, self.exploration))
        return (
            weights.inherited * item.inherited_score
            + (1.0 - exploration_amount) * relevance
            + exploration_amount * exploration
        )

    def score_image(self, candidate, page_title: str, page_score: float) -> float:
        relevance = self.relevance(candidate.url, candidate.alt, candidate.context, page_title)
        exploration = self.weights.noise * self.noise()
        exploration_amount = max(0.0, min(1.0, self.exploration))
        return (
            (1.0 - exploration_amount) * relevance
            + exploration_amount * exploration
            + self.weights.image_page * page_score
        )
