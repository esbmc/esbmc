from typing import Literal

NAME = "foo"


def foo(s: Literal[NAME]) -> int:
    return 42


assert foo("foo") == 42
