from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Sequence


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _wilson_interval(
    successes: int, total: int, z: float = 1.64
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = estimate + z * z / (2.0 * total)
    spread = z * math.sqrt(
        estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
    )
    return (
        _clamp((center - spread) / denominator),
        _clamp((center + spread) / denominator),
    )


@dataclass(frozen=True)
class RiskBlock:
    lower_mismatch: float
    upper_mismatch: float
    count: int
    invalid_count: int
    risk: float
    wilson_lower: float
    wilson_upper: float


@dataclass(frozen=True)
class MonotoneRiskMap:
    """Piecewise-constant empirical risk map estimated by PAV."""

    blocks: tuple[RiskBlock, ...]

    def lookup(self, raw_mismatch: float) -> float:
        value = _clamp(raw_mismatch)
        for block in self.blocks:
            if value <= block.upper_mismatch + 1e-12:
                return block.risk
        return self.blocks[-1].risk if self.blocks else 0.0

    def interval_width(self, raw_mismatch: float) -> float:
        value = _clamp(raw_mismatch)
        for block in self.blocks:
            if value <= block.upper_mismatch + 1e-12:
                return _clamp(block.wilson_upper - block.wilson_lower)
        if not self.blocks:
            return 1.0
        block = self.blocks[-1]
        return _clamp(block.wilson_upper - block.wilson_lower)


def fit_monotone_risk_map(
    raw_mismatch: Sequence[float],
    invalid_labels: Sequence[int],
    *,
    max_initial_blocks: int = 6,
) -> MonotoneRiskMap:
    """Fit a binned monotone invalid-reuse map with pool-adjacent violators."""

    if len(raw_mismatch) != len(invalid_labels):
        raise ValueError("raw_mismatch and invalid_labels must have equal length")
    if max_initial_blocks <= 0:
        raise ValueError("max_initial_blocks must be positive")
    if not raw_mismatch:
        return MonotoneRiskMap((RiskBlock(0.0, 1.0, 0, 0, 0.0, 0.0, 1.0),))

    pairs = sorted(
        (
            (_clamp(mismatch), int(bool(label)))
            for mismatch, label in zip(raw_mismatch, invalid_labels)
        ),
        key=lambda pair: pair[0],
    )
    score_groups: list[dict[str, float | int]] = []
    for mismatch, label in pairs:
        if score_groups and math.isclose(
            float(score_groups[-1]["lower"]),
            mismatch,
            abs_tol=1e-12,
        ):
            score_groups[-1]["count"] = int(score_groups[-1]["count"]) + 1
            score_groups[-1]["invalid"] = int(score_groups[-1]["invalid"]) + label
        else:
            score_groups.append(
                {
                    "lower": mismatch,
                    "upper": mismatch,
                    "count": 1,
                    "invalid": label,
                }
            )

    block_count = min(max_initial_blocks, len(score_groups))
    target_count = len(pairs) / block_count
    initial: list[dict[str, float | int]] = []
    current: list[dict[str, float | int]] = []
    current_count = 0
    for group_index, group in enumerate(score_groups):
        current.append(group)
        current_count += int(group["count"])
        blocks_left = block_count - len(initial) - 1
        groups_left = len(score_groups) - group_index - 1
        should_close = (
            blocks_left > 0
            and groups_left >= blocks_left
            and (current_count >= target_count or groups_left == blocks_left)
        )
        if not should_close:
            continue
        invalid_count = sum(int(item["invalid"]) for item in current)
        initial.append(
            {
                "lower": float(current[0]["lower"]),
                "upper": float(current[-1]["upper"]),
                "count": current_count,
                "invalid": invalid_count,
                "risk": invalid_count / current_count,
            }
        )
        current = []
        current_count = 0

    if current:
        invalid_count = sum(int(item["invalid"]) for item in current)
        initial.append(
            {
                "lower": float(current[0]["lower"]),
                "upper": float(current[-1]["upper"]),
                "count": current_count,
                "invalid": invalid_count,
                "risk": invalid_count / current_count,
            }
        )

    merged: list[dict[str, float | int]] = []
    for block in initial:
        merged.append(block)
        while len(merged) >= 2 and float(merged[-2]["risk"]) > float(
            merged[-1]["risk"]
        ):
            right = merged.pop()
            left = merged.pop()
            count = int(left["count"]) + int(right["count"])
            invalid_count = int(left["invalid"]) + int(right["invalid"])
            merged.append(
                {
                    "lower": float(left["lower"]),
                    "upper": float(right["upper"]),
                    "count": count,
                    "invalid": invalid_count,
                    "risk": invalid_count / count,
                }
            )

    output: list[RiskBlock] = []
    for block in merged:
        count = int(block["count"])
        invalid_count = int(block["invalid"])
        lower, upper = _wilson_interval(invalid_count, count)
        output.append(
            RiskBlock(
                lower_mismatch=float(block["lower"]),
                upper_mismatch=float(block["upper"]),
                count=count,
                invalid_count=invalid_count,
                risk=_clamp(float(block["risk"])),
                wilson_lower=lower,
                wilson_upper=upper,
            )
        )
    return MonotoneRiskMap(tuple(output))


@dataclass(frozen=True)
class CalibratedSignals:
    risk: float
    uncertainty: float
    calibration_uncertainty: float


def calibrate_signals(
    raw_mismatch: float,
    matching_uncertainty: float,
    risk_map: MonotoneRiskMap,
) -> CalibratedSignals:
    """Compute the uncertainty-weighted routing score D and final U."""

    mismatch = _clamp(raw_mismatch)
    calibration_uncertainty = risk_map.interval_width(mismatch)
    mapped_risk = risk_map.lookup(mismatch)
    risk = (
        1.0 - calibration_uncertainty
    ) * mapped_risk + calibration_uncertainty * mismatch
    return CalibratedSignals(
        risk=_clamp(risk),
        uncertainty=max(_clamp(matching_uncertainty), calibration_uncertainty),
        calibration_uncertainty=calibration_uncertainty,
    )


class Action(str, Enum):
    REUSE = "reuse"
    VERIFY = "verify"
    REGENERATE = "regenerate"


@dataclass(frozen=True)
class RoutingSignals:
    similarity: float
    risk: float
    uncertainty: float
    conflict: bool


@dataclass(frozen=True)
class RoutingPolicy:
    tau_reuse: float
    delta_reuse: float
    upsilon_reuse: float
    tau_verify: float
    tau_conflict_verify: float
    delta_verify: float
    upsilon_verify: float
    regenerate_only: bool = False

    def __post_init__(self) -> None:
        values = (
            self.tau_reuse,
            self.delta_reuse,
            self.upsilon_reuse,
            self.tau_verify,
            self.tau_conflict_verify,
            self.delta_verify,
            self.upsilon_verify,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("routing boundaries must lie in [0, 1]")
        if self.tau_verify > self.tau_reuse:
            raise ValueError("tau_verify must not exceed tau_reuse")
        if self.tau_conflict_verify < self.tau_verify:
            raise ValueError(
                "conflict verification must use a stricter similarity boundary"
            )
        if self.delta_verify < self.delta_reuse:
            raise ValueError("delta_verify must not be lower than delta_reuse")
        if self.upsilon_verify < self.upsilon_reuse:
            raise ValueError("upsilon_verify must not be lower than upsilon_reuse")

    @classmethod
    def regeneration_policy(cls) -> RoutingPolicy:
        return cls(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, regenerate_only=True)


@dataclass
class VerifierBudget:
    allowance: int
    used: int = 0

    @classmethod
    def from_rate(cls, rate: float, window_size: int) -> VerifierBudget:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must lie in [0, 1]")
        if window_size < 0:
            raise ValueError("window_size must be nonnegative")
        return cls(allowance=math.floor(rate * window_size))

    def consume(self) -> bool:
        if self.used >= self.allowance:
            return False
        self.used += 1
        return True


def route_candidate(
    signals: RoutingSignals,
    policy: RoutingPolicy,
    budget: VerifierBudget,
) -> Action:
    """Choose reuse, verification, or regeneration for one candidate."""

    if policy.regenerate_only:
        return Action.REGENERATE
    if (
        not signals.conflict
        and signals.similarity >= policy.tau_reuse
        and signals.risk <= policy.delta_reuse
        and signals.uncertainty <= policy.upsilon_reuse
    ):
        return Action.REUSE

    verify_boundary = (
        policy.tau_conflict_verify if signals.conflict else policy.tau_verify
    )
    in_verification_region = (
        signals.similarity >= verify_boundary
        and signals.risk <= policy.delta_verify
        and signals.uncertainty <= policy.upsilon_verify
    )
    if in_verification_region and budget.consume():
        return Action.VERIFY
    return Action.REGENERATE


def resolve_verification(action: Action, verifier_output: str | None) -> Action:
    """Return reuse only for an exact valid verifier decision."""

    if action is not Action.VERIFY:
        return action
    return Action.REUSE if verifier_output == "VALID" else Action.REGENERATE


@dataclass(frozen=True)
class CalibrationExample:
    signals: RoutingSignals
    reuse_is_valid: bool
    verifier_accepts: bool


@dataclass(frozen=True)
class PolicySelection:
    policy: RoutingPolicy
    feasible: bool
    safe_accepted: int
    unsafe_accepted: int
    verifier_calls: int


def _evaluate_policy(
    policy: RoutingPolicy,
    examples: Sequence[CalibrationExample],
    budget_rate: float,
) -> tuple[int, int, int]:
    budget = VerifierBudget.from_rate(budget_rate, len(examples))
    safe_accepted = 0
    unsafe_accepted = 0
    for example in examples:
        action = route_candidate(example.signals, policy, budget)
        accepted = action is Action.REUSE or (
            action is Action.VERIFY and example.verifier_accepts
        )
        if accepted and example.reuse_is_valid:
            safe_accepted += 1
        elif accepted:
            unsafe_accepted += 1
    return safe_accepted, unsafe_accepted, budget.used


def select_routing_policy(
    candidates: Iterable[RoutingPolicy],
    examples: Sequence[CalibrationExample],
    *,
    alpha: float,
    budget_rate: float,
) -> PolicySelection:
    """Select a feasible policy with an unweighted lexicographic objective."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if not 0.0 <= budget_rate <= 1.0:
        raise ValueError("budget_rate must lie in [0, 1]")

    feasible: list[
        tuple[tuple[int, int, int], RoutingPolicy, tuple[int, int, int]]
    ] = []
    for policy in candidates:
        safe_accepted, unsafe_accepted, verifier_calls = _evaluate_policy(
            policy,
            examples,
            budget_rate,
        )
        total_accepted = safe_accepted + unsafe_accepted
        accepted_unsafe_rate = (
            unsafe_accepted / total_accepted if total_accepted else 0.0
        )
        verifier_rate = verifier_calls / len(examples) if examples else 0.0
        if accepted_unsafe_rate <= alpha and verifier_rate <= budget_rate:
            objective = (
                safe_accepted,
                -unsafe_accepted,
                -verifier_calls,
            )
            feasible.append(
                (
                    objective,
                    policy,
                    (safe_accepted, unsafe_accepted, verifier_calls),
                )
            )

    if not feasible:
        return PolicySelection(
            policy=RoutingPolicy.regeneration_policy(),
            feasible=False,
            safe_accepted=0,
            unsafe_accepted=0,
            verifier_calls=0,
        )

    _, policy, metrics = max(feasible, key=lambda item: item[0])
    safe_accepted, unsafe_accepted, verifier_calls = metrics
    return PolicySelection(
        policy=policy,
        feasible=True,
        safe_accepted=safe_accepted,
        unsafe_accepted=unsafe_accepted,
        verifier_calls=verifier_calls,
    )
