# Membership on None in a later and/or operand raises TypeError (#8249). No
# guard can be planted there (#8275), so the TypeError is missed.
from typing import Optional


def has_item(xs: Optional[list], flag: bool) -> bool:
    return flag or 1 in xs


def main() -> None:
    has_item(None, False)


main()
