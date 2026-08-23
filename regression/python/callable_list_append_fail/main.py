# The stored callable is really invoked, not assumed.
from typing import Callable, List


def inc(m: int) -> int:
    return m + 1


def main() -> None:
    fs: List[Callable[[int], int]] = []
    fs.append(inc)
    assert fs[0](1) == 3


main()
