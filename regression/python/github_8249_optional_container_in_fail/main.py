# Membership on Optional[dict]/Optional[list] tests the container (#8249).
from typing import Optional


def has_key(classes: Optional[dict], tag: str) -> bool:
    return tag in classes


def main() -> None:
    classes: Optional[dict] = None
    if nondet_bool():
        classes = {"pre": "x"}
        assert has_key(classes, "post")


main()
