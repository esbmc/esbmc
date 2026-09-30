# str() of a runtime float renders CPython's repr where it can be proved.
def f(neg_zero: float, tenth: float, big: float, frac: float) -> None:
    assert str(neg_zero) == "-0.0"
    assert str(tenth) == "0.1"
    assert f"{big}" == "1000000000000000.0"
    assert str(frac) == "123456.789"


f(-0.0, 0.1, 1e15, 123456.789)
