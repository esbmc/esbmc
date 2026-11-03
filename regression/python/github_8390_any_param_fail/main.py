from typing import Any


def m(hint) -> bool:
    return hint is Any


assert not m(int)
assert not m(Any)
