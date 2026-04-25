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


def fields(class_or_instance):
    target = class_or_instance
    if not isinstance(class_or_instance, type):
        target = class_or_instance.__class__
    return getattr(target, "__dataclass_fields__", ())


def asdict(obj):
    return {field_name: getattr(obj, field_name) for field_name in fields(obj)}


def astuple(obj):
    return tuple(getattr(obj, field_name) for field_name in fields(obj))
