from typing import Callable, List


def inc(x: int) -> int:
    return x + 1


def dec(x: int) -> int:
    return x - 1


def main() -> None:
    fs: List[Callable[[int], int]] = [inc, dec]
    assert fs[0](1) == 2


main()
