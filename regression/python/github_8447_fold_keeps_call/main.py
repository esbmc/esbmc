calls: int = 0


def f() -> None:
    global calls
    calls += 1


def g() -> str:
    global calls
    calls += 10
    return "a"


def main() -> None:
    assert f() != 5
    assert g() != 1
    assert not (1 == g())
    assert calls == 21


main()
