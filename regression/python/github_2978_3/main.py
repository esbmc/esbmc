from typing import Literal, overload


class Foo:

    def __init__(self) -> None:
        pass


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
        obj = Foo()
        assert isinstance(obj, Foo), "Assertion: obj is a Foo instance"
        return obj
    elif s == "bar":
        obj = Bar()
        return obj
    else:
        raise ValueError("Invalid class name")


# Example call that will trigger the assertion failure
create("foo")
