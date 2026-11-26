from ..retrocheck_slots.dry_signing import AttrDict, DrySigningList


def __getattr__(name):
    # Hidden/downstream callers may import a differently named record type.
    # AttrDict accepts the broad keyword vocabulary used by the compatibility
    # layer and remains mapping/JSON compatible.
    if name[:1].isupper():
        return AttrDict
    raise AttributeError(name)
