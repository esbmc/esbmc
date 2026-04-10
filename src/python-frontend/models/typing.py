def TypeVar(name, *args, **kwargs) -> type:
    return object


class Any:

    def __class_getitem__(cls, item):
        return cls


class Callable:

    def __class_getitem__(cls, item):
        return cls


class Dict:

    def __class_getitem__(cls, item):
        return cls


class Generic:

    def __class_getitem__(cls, item):
        return cls


class Iterable:

    def __class_getitem__(cls, item):
        return cls


class Iterator:

    def __class_getitem__(cls, item):
        return cls


class List:

    def __class_getitem__(cls, item):
        return cls


class Optional:

    def __class_getitem__(cls, item):
        return cls


class Set:

    def __class_getitem__(cls, item):
        return cls


class Sized:

    def __class_getitem__(cls, item):
        return cls


class Tuple:

    def __class_getitem__(cls, item):
        return cls


class Type:

    def __class_getitem__(cls, item):
        return cls


class Union:

    def __class_getitem__(cls, item):
        return cls


# Abstract I/O types
class IO:
    pass


class BinaryIO(IO):
    pass


class TextIO(IO):
    pass
