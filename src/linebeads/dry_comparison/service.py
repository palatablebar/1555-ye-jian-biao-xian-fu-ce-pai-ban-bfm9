from ..retrocheck_slots.dry_signing import *  # noqa: F403
from ..retrocheck_slots.dry_signing import universal_attribute


def __getattr__(name):
    return universal_attribute(name)
