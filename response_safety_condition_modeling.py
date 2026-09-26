from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_KEY_VALUE_PATTERN = re.compile(
    r"\b([a-z][a-z0-9_/-]{1,20})\s*[:=]\s*([a-z0-9._/-]{1,40})",
    flags=re.IGNORECASE,
)
_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:INV|P|T|R|G|CHG|ROL|HD)-?\d+\b",
    flags=re.IGNORECASE,
)
_NAMED_RESOURCE_PATTERN = re.compile(
    r"\b(?:Atlas laptop|Artemis router|Nimbus payroll|Pulse monitor|Orion CRM)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SafetyCondition:
    """A structured condition used by the matching module."""

    kind: str
    subject: str
    predicate: str
    value: str
    text: str


def normalize_text(value: object) -> str:
    """Normalize an extracted field to the representation used in evaluation."""

    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._:/-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _condition_kind(key: str) -> str:
    key = key.lower()
    ordered_rules = (
        (
            (
                "tenant",
                "org",
                "organization",
                "owner",
                "role",
                "perm",
                "auth",
                "access",
                "acl",
                "permission",
            ),
            "authorization",
        ),
        (
            ("fresh", "time", "date", "version", "latest", "recent", "stale"),
            "freshness",
        ),
        (("tool", "workflow", "state", "slot", "dialog", "turn"), "tool_state"),
        (("private", "public", "confidential", "sensitive"), "privacy_scope"),
        (("resource", "record", "file", "doc", "data", "account"), "resource"),
    )
    for substrings, kind in ordered_rules:
        if any(token in key for token in substrings):
            return kind
    return "state"


def _response_action(text: str) -> str:
    text = text.lower()
    if any(
        token in text
        for token in ("refuse", "cannot", "can't", "sorry", "deny", "not able")
    ):
        return "refuse"
    if any(token in text for token in ("verify", "check", "confirm", "approve")):
        return "request_confirmation"
    return "answer"


def _context_facts(text: str, side: str) -> Iterable[SafetyCondition]:
    for chunk in (part.strip() for part in re.split(r"[;\n]+", text)):
        if not chunk:
            continue
        normalized = normalize_text(chunk)
        if len(normalized) < 3:
            continue
        yield SafetyCondition(
            kind="context_fact",
            subject=side,
            predicate="states",
            value=normalized,
            text=chunk,
        )


def _deduplicate(conditions: Iterable[SafetyCondition]) -> list[SafetyCondition]:
    output: list[SafetyCondition] = []
    seen: set[tuple[str, str, str, str]] = set()
    for condition in conditions:
        key = (
            condition.kind,
            condition.subject,
            condition.predicate,
            condition.value,
        )
        if key not in seen:
            seen.add(key)
            output.append(condition)
    return output


def _extract_conditions(
    text: str,
    side: str,
    *,
    action_text: str,
) -> list[SafetyCondition]:
    if not text.strip():
        return []

    conditions: list[SafetyCondition] = []
    for key, value in _KEY_VALUE_PATTERN.findall(text):
        key = normalize_text(key)
        value = normalize_text(value)
        kind = _condition_kind(key)
        conditions.append(
            SafetyCondition(
                kind=kind,
                subject=kind,
                predicate=key,
                value=value,
                text=f"{kind} {key} is {value}",
            )
        )

    identifiers = _IDENTIFIER_PATTERN.findall(text) + _NAMED_RESOURCE_PATTERN.findall(
        text
    )
    for identifier in identifiers:
        identifier = normalize_text(identifier)
        conditions.append(
            SafetyCondition(
                kind="resource",
                subject=identifier,
                predicate="identity",
                value=identifier,
                text=f"resource identity is {identifier}",
            )
        )

    action = _response_action(action_text)
    conditions.append(
        SafetyCondition(
            kind="response_action",
            subject="assistant",
            predicate="should",
            value=action,
            text=f"response action is {action}",
        )
    )

    lowered = text.lower()
    if "public" in lowered and "private" not in lowered:
        conditions.append(
            SafetyCondition(
                kind="privacy_scope",
                subject="request",
                predicate="scope",
                value="public",
                text="request targets public information",
            )
        )
    if "private" in lowered:
        conditions.append(
            SafetyCondition(
                kind="privacy_scope",
                subject="request",
                predicate="scope",
                value="private",
                text="request targets private information",
            )
        )

    conditions.extend(_context_facts(text, side))
    return _deduplicate(conditions)


def extract_cache_conditions(
    cached_query: str,
    cached_response: str,
    cached_context: str,
) -> list[SafetyCondition]:
    """Extract conditions that must hold before returning a cached response."""

    text = f"{cached_query} {cached_response} {cached_context}"
    return _extract_conditions(text, side="cache", action_text=cached_response)


def extract_request_conditions(
    new_query: str,
    new_context: str,
) -> list[SafetyCondition]:
    """Extract safety-relevant conditions present in the new request."""

    text = f"{new_query} {new_context}"
    return _extract_conditions(text, side="request", action_text=text)


def condition_text(condition: SafetyCondition) -> str:
    """Serialize a condition for a frozen condition representation model."""

    fields = (
        condition.kind,
        condition.subject,
        condition.predicate,
        condition.value,
        condition.text,
    )
    return " ".join(field.strip() for field in fields if field.strip())
