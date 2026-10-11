# Membership on Optional[dict]/Optional[list] tests the container (#8249).
from typing import Optional


def has_key(classes: Optional[dict], tag: str) -> bool:
    return tag in classes


def has_item(xs: Optional[list], x: int) -> bool:
    return x in xs


def main() -> None:
    classes: Optional[dict] = None
    if nondet_bool():
        classes = {"pre": "x"}
        assert has_key(classes, "pre")
        assert not has_key(classes, "post")
        assert "post" not in classes
    try:
        has_key(classes, "pre")
    except TypeError:
        assert classes is None
    assert has_item([1, 2], 2)
    assert not has_item([1, 2], 3)


main()
