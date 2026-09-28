# The old model proved str(1e-05) differs from CPython's "1e-05"; the repr needs
# scientific notation or more than 6 fractional digits, so it is not rendered.
def f(v: float) -> None:
    assert str(v) != "1e-05"


f(1e-05)
