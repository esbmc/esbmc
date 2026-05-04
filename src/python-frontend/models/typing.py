# pylint: disable=function-redefined,unused-argument
# Operational-model stubs: stdlib shadows for ESBMC models. Argument
# names on `__class_getitem__` and `TypeVar` are part of the API contract
# matched by ESBMC's Python converter, even when the abstract body does
# not reference them.
def TypeVar(name, *args, **kwargs) -> type:
    return object


class Any:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class Callable:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class ClassVar:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class Dict:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class Generic:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class Iterable:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class Iterator:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class List:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class Optional:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class Set:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class Sized:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class Tuple:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class Type:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


class ForwardRef:
    pass


class Union:

    def __class_getitem__(cls, item):
        """Return cls for generic-alias subscription support."""
        return cls


# Abstract I/O types
class IO:
    pass


class BinaryIO(IO):
    pass


class TextIO(IO):
    pass
