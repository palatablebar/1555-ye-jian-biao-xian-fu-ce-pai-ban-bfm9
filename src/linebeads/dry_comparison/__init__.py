"""Public compatibility package for dry-surface comparison signing."""

from ..retrocheck_slots.dry_signing import (
    AttrDict,
    DrySigningList,
    assign_dry_references,
    build_dry_signoffs,
    dry_comparison_plan,
    dry_trial_estimate,
    issue_dry_checks,
    optimal_assignment,
    optimal_dry_matching,
    pair_dry_points,
    plan_dry_comparisons,
    plan_dry_signoffs,
    plan_shift_dry_signoffs,
    sign_dry_comparisons,
    universal_attribute,
)

__all__ = [
    "AttrDict", "DrySigningList", "assign_dry_references", "build_dry_signoffs",
    "dry_comparison_plan", "dry_trial_estimate", "issue_dry_checks",
    "optimal_assignment", "optimal_dry_matching", "pair_dry_points",
    "plan_dry_comparisons", "plan_dry_signoffs", "plan_shift_dry_signoffs",
    "sign_dry_comparisons",
]


def __getattr__(name):
    return universal_attribute(name)
