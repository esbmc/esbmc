from typing import Literal


def foo(s: Literal[b"foo"]) -> int:
    return 42


foo(b"foo")
