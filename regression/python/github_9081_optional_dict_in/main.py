# Membership on an Optional[dict[...]] tests the dict it holds, and raises
# TypeError when it holds None (#9081).
from typing import Optional


def main() -> None:
    o: Optional[dict[str, str]] = None
    if nondet_bool():
        e: dict[str, str] = {"a": "b"}
        o = e
    if o is not None:
        assert "a" in o
        assert "c" not in o
    try:
        assert "a" in o
    except TypeError:
        assert o is None

    extras: dict[str, Optional[dict[str, str]]] = {}
    extras["k"] = None
    h = extras["k"]
    raised = False
    try:
        "t" in h
    except TypeError:
        raised = True
    assert raised


main()
