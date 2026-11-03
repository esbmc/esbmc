calls: int = 0


def g() -> int:
    global calls
    calls += 1
    return calls


def main() -> None:
    b: bool = 0 < g() < 5
    assert b
    assert calls == 2


main()
