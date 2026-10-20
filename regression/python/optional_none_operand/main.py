from typing import Optional


def inc(v: Optional[int]) -> int:
    if v is None:
        return 0
    return v + 1


def at_least(n: int, limit: Optional[int]) -> bool:
    if limit is None:
        return True
    return n >= limit


def main() -> None:
    v: Optional[int] = None
    k: int = 4
    if nondet_bool():
        v = k
    if v is not None:
        w = v * 2
        assert w == 8
    assert inc(None) == 0
    assert inc(2) == 3
    assert at_least(3, None)
    assert not at_least(3, 5)
    x: Optional[int] = 3
    assert x + 1 == 4


main()
