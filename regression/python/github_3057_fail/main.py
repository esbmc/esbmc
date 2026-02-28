from typing import Literal, overload


class Helper:

    def __init__(self) -> None:
        pass


class Foo:

    def __init__(self) -> None:
        self.h = Helper()


class Bar:

    def __init__(self) -> None:
        pass


@overload
def create(s: Literal["foo"]) -> Foo:
    ...


@overload
def create(s: Literal["bar"]) -> Bar:
    ...


def create(s: str) -> Foo | Bar:
    if s == "foo":
        return Foo()
    elif s == "bar":
        return Bar()
    else:
        raise ValueError("Invalid class name")


b = create("bar")

assert isinstance(b, Foo)
