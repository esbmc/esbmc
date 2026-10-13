from typing import Any


def used() -> int:
    return 1


def unused(h) -> bool:
    return h is Any


assert used() == 1
