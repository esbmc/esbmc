from typing import Literal


def foo(s: Literal["foo"]) -> int:
    return 42


assert foo("foo") == 41
