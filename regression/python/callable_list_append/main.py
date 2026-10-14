# A function name stored in a list must decay to a function pointer, as it
# already does in call-argument position. Storing the code symbol itself
# aborted conversion with "got invalid code for function". Issue #6640.
from typing import Callable, List


def inc(m: int) -> int:
    return m + 1


def dbl(m: int) -> int:
    return m * 2


class Bus:
    def __init__(self) -> None:
        self.subs: List[Callable[[int], int]] = []


def main() -> None:
    fs: List[Callable[[int], int]] = []
    fs.append(inc)
    assert fs[0](1) == 2

    # The stored callables stay distinct and dispatch by index.
    fs.append(dbl)
    assert len(fs) == 2
    assert fs[0](3) == 4
    assert fs[1](3) == 6

    # A list held in an instance field accepts one too.
    b = Bus()
    b.subs.append(inc)
    assert len(b.subs) == 1


main()
