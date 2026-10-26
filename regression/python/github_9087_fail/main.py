# GitHub #9087: indexing a Sequence[str] parameter out of range raises.
from typing import Sequence


def second(xs: Sequence[str]) -> str:
    return xs[1]


def main() -> None:
    second(["a"])


main()
