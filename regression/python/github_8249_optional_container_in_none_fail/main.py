# Membership on a None Optional[dict]/Optional[list] raises TypeError (#8249).
from typing import Optional


def has_item(xs: Optional[list], x: int) -> bool:
    return x in xs


def main() -> None:
    xs: Optional[list] = None
    if nondet_bool():
        xs = [1, 2]
    has_item(xs, 1)


main()
