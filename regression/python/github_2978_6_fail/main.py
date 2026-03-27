from typing import Literal


def foo(s: Literal["foo", 42, True]) -> int:
    return 42


assert foo("foo") == 41
