# pylint: disable=unused-argument
# Operational-model stubs for dataclasses module.


class Field:

    def __init__(self, name):
        self.name = name


class InitVar:

    @classmethod
    def __class_getitem__(cls, item):
        return cls


def dataclass(_cls=None, **kwargs):

    def wrap(cls):
        annotations = getattr(cls, "__annotations__", {})
        cls.__dataclass_fields__ = [
            Field(name)
            for name, annotation in annotations.items()
            if not _is_classvar_annotation(annotation)
            and not _is_initvar_annotation(annotation)
        ]
        return cls

    if _cls is None:
        return wrap
    return wrap(_cls)


def field(*args, default=None, default_factory=None, **kwargs):
    if default_factory is not None:
        return default_factory()
    return default


def is_dataclass(obj):
    if isinstance(obj, type):
        return hasattr(obj, "__dataclass_fields__")
    return hasattr(type(obj), "__dataclass_fields__")


def fields(obj):
    if not is_dataclass(obj):
        raise TypeError("fields() should be called on dataclass types or instances")
    target_cls = obj if isinstance(obj, type) else type(obj)
    return list(getattr(target_cls, "__dataclass_fields__", []))


def asdict(obj):
    if not is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    if isinstance(obj, type):
        raise TypeError("asdict() should be called on dataclass instances")
    return _convert(obj, "dict")


def astuple(obj):
    if not is_dataclass(obj):
        raise TypeError("astuple() should be called on dataclass instances")
    if isinstance(obj, type):
        raise TypeError("astuple() should be called on dataclass instances")
    return _convert(obj, "tuple")


def replace(obj, **changes):
    if not is_dataclass(obj):
        raise TypeError("replace() should be called on dataclass instances")
    if isinstance(obj, type):
        raise TypeError("replace() should be called on dataclass instances")

    field_names = [field_obj.name for field_obj in fields(obj)]
    for key in changes:
        if key not in field_names:
            raise TypeError("replace() got an unexpected field name")

    values = {name: getattr(obj, name) for name in field_names}
    values.update(changes)
    try:
        return type(obj)(**values)
    except TypeError:
        ordered_values = [values[name] for name in field_names]
        return type(obj)(*ordered_values)


def _is_initvar_annotation(annotation):
    return annotation is InitVar


def _is_classvar_annotation(annotation):
    if annotation is None:
        return False
    if getattr(annotation, "__name__", None) == "ClassVar":
        return True
    origin = getattr(annotation, "__origin__", None)
    return getattr(origin, "__name__", None) == "ClassVar"


def _convert(value, mode):
    if is_dataclass(value) and not isinstance(value, type):
        if mode == "dict":
            return {
                field_obj.name: _convert(getattr(value, field_obj.name), mode)
                for field_obj in fields(value)
            }
        return tuple(
            _convert(getattr(value, field_obj.name), mode)
            for field_obj in fields(value)
        )

    if isinstance(value, list):
        return [_convert(item, mode) for item in value]
    if isinstance(value, tuple):
        return tuple(_convert(item, mode) for item in value)
    if isinstance(value, dict):
        return {key: _convert(item, mode) for key, item in value.items()}

    return value
