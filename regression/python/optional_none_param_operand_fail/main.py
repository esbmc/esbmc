from typing import Optional


def at_least(n: int, limit: Optional[int]) -> bool:
    return n >= limit


def main() -> None:
    limit: Optional[int] = None
    if nondet_bool():
        limit = 5
    at_least(3, limit)


main()
