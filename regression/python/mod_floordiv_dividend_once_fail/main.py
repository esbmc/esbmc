calls = 0


def f() -> int:
    global calls
    calls += 1
    return -7


def main() -> None:
    q = f() // 5
    assert q == -2
    assert calls > 1


main()
