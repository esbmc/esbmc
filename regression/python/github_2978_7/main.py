from typing import Literal

def foo(s: Literal[Literal["foo"]]) -> int:
    return 42

assert foo("foo") == 42
