# pylint: disable=unused-argument
# Operational-model stubs: argument names are part of the API contract
# matched by ESBMC's Python converter, even when the body does not
# reference them.


class Field:

    def __class_getitem__(cls, item):
        """Return the class itself for subscription-style type usage."""
        return cls


class InitVar:

    def __class_getitem__(cls, item):
        """Return the class itself for subscription-style type usage."""
        return cls


def dataclass(_cls=None, **kwargs):

    def wrap(cls):
        return cls

    if _cls is None:
        return wrap
    return _cls


def field(*args, default=None, default_factory=None, **kwargs):
    if default_factory is not None:
        return default_factory()
    return default
