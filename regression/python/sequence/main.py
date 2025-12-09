from typing import Sequence

def foo(s: Sequence[str] | None = None) -> None:
    if s is None:
        return

foo(None)
