from typing import Literal


def foo(s: Literal["bar", "baz"] | None) -> None:
    assert s is not None
    if s == "bar":
        pass
    else:
        assert s == "baz"


foo("bar")
