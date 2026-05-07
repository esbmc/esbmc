# pylint: disable=unused-argument
# Operational-model stubs for dataclasses module.


class Field:

    def __init__(self, name):
        self.name = name


class InitVar:
    pass


def dataclass(_cls=None, **kwargs):

    def wrap(cls):
        return cls

    if _cls is None:
        return wrap
    return wrap(_cls)


def field(*args, default=None, default_factory=None, **kwargs):
    if default_factory is not None:
        return default_factory()
    return default


def is_dataclass(obj):
    return hasattr(obj, "__dataclass_fields__")


def fields(obj):
    if not is_dataclass(obj):
        raise TypeError("fields() should be called on dataclass types or instances")
    return []


def asdict(obj):
    if not is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return 0


def astuple(obj):
    if not is_dataclass(obj):
        raise TypeError("astuple() should be called on dataclass instances")
    return 0


def replace(obj, **changes):
    if not is_dataclass(obj):
        raise TypeError("replace() should be called on dataclass instances")
    return obj
