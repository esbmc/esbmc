from typing import Callable, List


def inc(x: int) -> int:
    return x + 1


def main() -> None:
    fs: List[Callable[[int], int]] = [inc]
    assert fs[0](1) == 3


main()
