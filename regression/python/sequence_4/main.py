from typing import Sequence


def foo(s: Sequence[str] | None = None) -> None:
    if len(s) == 0:
        return

    for x in s:
        assert isinstance(x, str)


foo(["", "test", "123"])
