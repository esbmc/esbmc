from typing import Sequence


def foo(s: Sequence[str] | None = None) -> None:
    if s is not None:
        return

    assert isinstance(s, str)


foo()
