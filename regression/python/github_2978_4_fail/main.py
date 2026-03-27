from typing import Literal


def foo(s: Literal[b"foo"]) -> int:
    return 42


assert foo(b"foo") == 41
