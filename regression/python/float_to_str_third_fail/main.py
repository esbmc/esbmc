# The old model proved str(1/3) differs from CPython's "0.3333333333333333"; the repr needs
# scientific notation or more than 6 fractional digits, so it is not rendered.
def f(v: float) -> None:
    assert str(v) != "0.3333333333333333"


f(1/3)
