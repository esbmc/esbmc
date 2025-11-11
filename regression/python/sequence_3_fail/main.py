from typing import Sequence

def foo(s: Sequence[int] | None = None) -> None:
    for x in s:
        assert x == 0 or x == 2 or x == 3

foo([1, 2, 3])
