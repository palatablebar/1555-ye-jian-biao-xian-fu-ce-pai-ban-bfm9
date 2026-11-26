"""retrocheck slots."""

from .service import plan_jobs, action_evidence
from .dry_signing import sign_dry_comparisons

__all__ = ('plan_jobs', 'action_evidence', 'sign_dry_comparisons')


def __getattr__(name):
    from .dry_signing import universal_attribute
    try:
        return universal_attribute(name)
    except AttributeError:
        raise AttributeError(name) from None
