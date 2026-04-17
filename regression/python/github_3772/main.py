def f() -> str:
    return "x"


def test() -> None:
    y: int = f()
    assert y == "x"


test()
