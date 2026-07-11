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


def host_of(url: str) -> str:
    return urlparse(url).hostname or ""


@dataclass
class ScoreWeights:
    host_freshness: float = 0.5
    noise: float = 0.25
    inherited: float = 1.0
    image_page: float = 0.2


class ScorePolicy:
    """Pure scorer. Depends only on a TopicState, an RNG, and a host->visits map."""

    def __init__(self, topic_state, rng, host_visits: dict, *, focus: float = 0.5, weights: ScoreWeights | None = None) -> None:
        self.topic_state = topic_state
        self.rng = rng
        self.host_visits = host_visits
        self.focus = focus
        self.weights = weights or ScoreWeights()

    def relevance(self, *texts: str) -> float:
        return self.topic_state.score_text(*texts)

    def host_freshness(self, url: str) -> float:
        return 1.0 / (1.0 + self.host_visits.get(host_of(url), 0))

    def noise(self) -> float:
        return self.rng.random()

    def score_link(self, item) -> float:
        weights = self.weights
        relevance = self.relevance(item.url, item.context)
        exploration = weights.host_freshness * self.host_freshness(item.url) + weights.noise * self.noise()
        return (
            weights.inherited * item.inherited_score
            + self.focus * relevance
            + (1.0 - self.focus) * exploration
        )

    def score_image(self, candidate, page_title: str, page_score: float) -> float:
        relevance = self.relevance(candidate.url, candidate.alt, candidate.context, page_title)
        exploration = self.weights.noise * self.noise()
        return (
            self.focus * relevance
            + (1.0 - self.focus) * exploration
            + self.weights.image_page * page_score
        )
