calls: int = 0


def g() -> int:
    global calls
    calls += 1
    return calls


def is_lower(c: str) -> bool:
    return 97 <= ord(c.lower()) <= 122


def main() -> None:
    assert is_lower("Q")
    assert not is_lower("1")

    global calls
    b: bool = 0 < g() < 5
    assert b
    assert calls == 1

    calls = 0
    c: bool = 0 < 1 <= g() < 5 < 9
    assert c
    assert calls == 1

    calls = 0
    d: bool = 3 < 1 < g()
    assert not d
    assert calls == 0

    calls = 0
    e: bool = 0 < g() < g() < g()
    assert e
    assert calls == 3

    calls = 0
    n: int = 0
    while 0 <= g() < 3:
        n += 1
    assert n == 2
    assert calls == 3


main()
