from typing import Literal

def foo(s: Literal["bar"] | None) -> None:
    assert s is None
    assert s != "bar"

foo("bar")
