from typing import Sequence

def foo(s: Sequence[str] | None = None) -> None:
    if s is None:
        return

    if len(s) == 0:
        return

    for x in s:
        assert isinstance(x, str)

foo(None)
foo([])
foo(["a", "b", "c"])
foo(["", "test", "123"])
