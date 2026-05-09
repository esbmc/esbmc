# pylint: disable=unused-argument
# Minimal operational-model stubs for dataclasses module


class InitVar:

    @classmethod
    def __class_getitem__(cls, item):
        return cls


class Field:

    def __init__(self, name):
        self.name = name


def dataclass(_cls=None, **kwargs):

    def wrap(cls):
        return cls

    if _cls is None:
        return wrap
    return wrap(_cls)


def field(*args, **kwargs):
    return None


def is_dataclass(obj):
    return False


def fields(obj):
    return []


def asdict(obj):
    if isinstance(obj, dict):
        raise TypeError("asdict() should be called on dataclass instances")
    return {}


def astuple(obj):
    return ()


def replace(obj, **changes):
    return obj
