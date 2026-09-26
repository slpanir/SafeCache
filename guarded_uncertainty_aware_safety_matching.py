from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import sqrt
import re
from statistics import mean
from typing import Iterable, Sequence

from response_safety_condition_modeling import SafetyCondition, condition_text


@dataclass(frozen=True)
class MatchingResult:
    """Signals produced for safety-conditioned cache routing."""

    raw_mismatch: float
    matching_uncertainty: float
    rule_conflict: bool
    context_conflict: bool
    best_support: tuple[float, ...]
    top_two_margins: tuple[float, ...]

    @property
    def conflict(self) -> bool:
        return self.rule_conflict or self.context_conflict


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return nonnegative cosine support in the interval [0, 1]."""

    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return _clamp(cosine)


def lexical_hash_embedding(text: str, dimension: int = 256) -> tuple[float, ...]:
    """Create the frozen signed lexical-hashing representation."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    vector = [0.0] * dimension
    for token in re.findall(r"[a-z0-9_-]+", text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return tuple(vector)


def condition_similarity(
    cached_condition: SafetyCondition,
    request_condition: SafetyCondition,
) -> float:
    """Compare two structured records with the frozen representation."""

    return cosine_similarity(
        lexical_hash_embedding(condition_text(cached_condition)),
        lexical_hash_embedding(condition_text(request_condition)),
    )


def top_two_margin_uncertainty(margins: Iterable[float]) -> float:
    """Compute the parameter-free ambiguity signal from top-two margins."""

    values = [_clamp(margin) for margin in margins]
    if not values:
        return 1.0
    return _clamp(1.0 - mean(values))


_OPPOSITE_VALUES = (
    (
        {"allow", "allowed", "approve", "approved", "grant", "granted", "true"},
        {"deny", "denied", "revoke", "revoked", "false"},
    ),
    (
        {"active", "available", "enabled", "open", "passed", "success", "valid"},
        {
            "inactive",
            "unavailable",
            "disabled",
            "closed",
            "failed",
            "failure",
            "invalid",
        },
    ),
    ({"public"}, {"private", "confidential", "sensitive"}),
    ({"answer"}, {"refuse"}),
)


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _values_conflict(left: str, right: str) -> bool:
    if left == right or "unknown" in {left, right}:
        return False
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    for positive, negative in _OPPOSITE_VALUES:
        if (left_tokens & positive and right_tokens & negative) or (
            left_tokens & negative and right_tokens & positive
        ):
            return True
    return False


def _equivalent_denials(left: str, right: str) -> bool:
    denial_terms = {
        "deny",
        "denied",
        "false",
        "not",
        "revoke",
        "revoked",
        "unauthorized",
    }
    return bool(_tokens(left) & denial_terms) and bool(_tokens(right) & denial_terms)


def explicit_condition_conflict(
    cached_conditions: Sequence[SafetyCondition],
    request_conditions: Sequence[SafetyCondition],
) -> bool:
    """Return a Boolean guard for explicit structured contradictions."""

    high_impact_kinds = {
        "authorization",
        "freshness",
        "privacy_scope",
        "resource",
        "response_action",
        "state",
        "tool_state",
    }
    for cached in cached_conditions:
        for request in request_conditions:
            same_field = (
                cached.kind == request.kind
                and cached.predicate == request.predicate
                and cached.subject == request.subject
            )
            if same_field and cached.kind in high_impact_kinds:
                changed = cached.value != request.value and "unknown" not in {
                    cached.value,
                    request.value,
                }
                if changed and not _equivalent_denials(cached.value, request.value):
                    return True
                if _values_conflict(cached.value, request.value):
                    return True

    cached_resources = {
        condition.value
        for condition in cached_conditions
        if condition.kind == "resource" and condition.value != "unknown"
    }
    request_resources = {
        condition.value
        for condition in request_conditions
        if condition.kind == "resource" and condition.value != "unknown"
    }
    return bool(
        cached_resources
        and request_resources
        and cached_resources.isdisjoint(request_resources)
    )


def explicit_context_conflict(
    cached_query: str,
    cached_context: str,
    new_query: str,
    new_context: str,
) -> bool:
    """Detect explicit source-input reversals not retained by extraction."""

    cached = f"{cached_query} {cached_context}".lower()
    current = f"{new_query} {new_context}".lower()
    cached_tokens = _tokens(cached)
    current_tokens = _tokens(current)
    pairs = (
        (
            ("granted", "allowed", "authorized", "owner"),
            ("denied", "revoked", "unauthorized"),
        ),
        (
            ("active", "enabled", "available", "passed"),
            ("inactive", "disabled", "unavailable", "failed"),
        ),
        (("public",), ("private", "confidential")),
    )
    for positive, negative in pairs:
        if cached_tokens.intersection(positive) and current_tokens.intersection(
            negative
        ):
            return True
        if cached_tokens.intersection(negative) and current_tokens.intersection(
            positive
        ):
            return True

    key_value_pattern = re.compile(
        r"\b([a-z][a-z0-9_/-]{1,20})\s*[:=]\s*([a-z0-9._/-]{1,40})",
        flags=re.IGNORECASE,
    )
    cached_fields = {
        key.lower(): value.lower() for key, value in key_value_pattern.findall(cached)
    }
    current_fields = {
        key.lower(): value.lower() for key, value in key_value_pattern.findall(current)
    }
    state_keys = (
        "access",
        "auth",
        "date",
        "fresh",
        "owner",
        "permission",
        "role",
        "state",
        "time",
        "tool",
        "version",
        "workflow",
    )
    for key in cached_fields.keys() & current_fields.keys():
        if (
            any(token in key for token in state_keys)
            and cached_fields[key] != current_fields[key]
        ):
            return True

    correction_cues = {
        "actually",
        "change",
        "changed",
        "correct",
        "instead",
        "now",
        "rather",
        "switch",
    }
    has_correction = bool(current_tokens.intersection(correction_cues))
    temporal_pattern = re.compile(
        r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b"
        r"|\b\d{1,2}:\d{2}(?::\d{2})?\b",
        flags=re.IGNORECASE,
    )
    cached_temporal = set(temporal_pattern.findall(cached))
    current_temporal = set(temporal_pattern.findall(current))
    temporal_change = (
        cached_temporal
        and current_temporal
        and (
            cached_temporal.isdisjoint(current_temporal)
            or (has_correction and cached_temporal != current_temporal)
        )
    )
    if temporal_change:
        return True

    if has_correction:
        identifier_pattern = re.compile(
            r"\b(?:INV|P|T|R|G|CHG|ROL|HD)-?\d+\b",
            flags=re.IGNORECASE,
        )
        cached_identifiers = {
            value.lower() for value in identifier_pattern.findall(cached)
        }
        current_identifiers = {
            value.lower() for value in identifier_pattern.findall(current)
        }
        if cached_identifiers and current_identifiers - cached_identifiers:
            return True
    return False


def match_safety_conditions(
    cached_conditions: Sequence[SafetyCondition],
    request_conditions: Sequence[SafetyCondition],
    *,
    cached_query: str,
    cached_context: str,
    new_query: str,
    new_context: str,
) -> MatchingResult:
    """Match every cached-response condition to its strongest request support."""

    context_conflict = explicit_context_conflict(
        cached_query,
        cached_context,
        new_query,
        new_context,
    )

    if not cached_conditions or not request_conditions:
        return MatchingResult(
            raw_mismatch=1.0,
            matching_uncertainty=1.0,
            rule_conflict=False,
            context_conflict=context_conflict,
            best_support=(),
            top_two_margins=(),
        )

    best_support: list[float] = []
    margins: list[float] = []
    for cached in cached_conditions:
        scores = sorted(
            (
                _clamp(condition_similarity(cached, request))
                for request in request_conditions
            ),
            reverse=True,
        )
        best = scores[0] if scores else 0.0
        second = scores[1] if len(scores) > 1 else 0.0
        best_support.append(best)
        margins.append(max(0.0, best - second))

    raw_mismatch = _clamp(mean(1.0 - score for score in best_support))
    return MatchingResult(
        raw_mismatch=raw_mismatch,
        matching_uncertainty=top_two_margin_uncertainty(margins),
        rule_conflict=explicit_condition_conflict(
            cached_conditions, request_conditions
        ),
        context_conflict=context_conflict,
        best_support=tuple(best_support),
        top_two_margins=tuple(margins),
    )


match_conditions = match_safety_conditions
