calls = 0


def f() -> int:
    global calls
    calls += 1
    return -7


def main() -> None:
    r = f() % 5
    assert calls == 1
    assert r == 3
    q = f() // 5
    assert calls == 2
    assert q == -2


main()
