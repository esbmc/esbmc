from typing import Sequence

def foo(s: Sequence[str] | None = None) -> None:
    assert False

foo(None)
