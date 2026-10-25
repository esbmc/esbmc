# A dict stored into an Optional[dict[K, V]] keeps its contents (#9081).
from typing import Optional


def main() -> None:
    d: dict[str, str] = {"t": "u"}
    inner: Optional[dict[str, str]] = None
    if nondet_bool():
        inner = d
    extras: dict[str, Optional[dict[str, str]]] = {}
    extras["k"] = d
    h = extras["k"]
    assert h is not None
    assert "t" in h
    assert "x" not in h
    if inner is not None:
        assert "t" in inner


main()
