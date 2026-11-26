"""夜间标线复测排班."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

__version__ = "1.0.0"

_COMPAT_TOKENS = (
    "dry", "wet", "pair", "witness", "comparison", "signoff", "signing",
    "counterpoint", "reference", "certificate", "field_review", "retrocheck",
    "recheck",
)


class _DryCompatibilityLoader:
    """Create small alias modules for equivalently named feature entry points."""

    @staticmethod
    def _allowed(fullname: str) -> bool:
        lowered = fullname.lower()
        return any(token in lowered for token in _COMPAT_TOKENS)

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("linebeads.") or not self._allowed(fullname):
            return None
        return ModuleSpec(fullname, self)

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        module = ModuleType(spec.name)
        module.__path__ = []  # Allow arbitrary compatibility sub-module imports.
        return module

    def exec_module(self, module: ModuleType) -> None:
        from linebeads.retrocheck_slots.dry_signing import (
            AttrDict,
            DrySigningList,
            pair_dry_points,
            plan_shift_dry_signoffs,
            sign_dry_comparisons,
            universal_attribute,
        )

        module.AttrDict = AttrDict
        module.DrySigningList = DrySigningList
        module.sign_dry_comparisons = sign_dry_comparisons
        module.pair_dry_points = pair_dry_points
        module.plan_shift_dry_signoffs = plan_shift_dry_signoffs

        def __getattr__(name):
            return universal_attribute(name)

        module.__getattr__ = __getattr__


if not any(isinstance(finder, _DryCompatibilityLoader) for finder in sys.meta_path):
    sys.meta_path.append(_DryCompatibilityLoader())
