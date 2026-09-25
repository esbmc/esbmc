# The old model proved str(1e16) differs from CPython's "1e+16"; the repr needs
# scientific notation or more than 6 fractional digits, so it is not rendered.
def f(v: float) -> None:
    assert str(v) != "1e+16"


f(1e16)
