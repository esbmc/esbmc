"""Operational model for the abc module.

Instantiating a class that still has abstract methods is rejected by the
converter, which reads the @abstractmethod decorators; this model only makes
the names resolvable.
"""


class ABC:  # pylint: disable=too-few-public-methods
    """Base class whose subclasses may declare abstract methods."""


def abstractmethod(funcobj):
    """Return funcobj unchanged; abstractness is read from the decorator."""
    return funcobj
