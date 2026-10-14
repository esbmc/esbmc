# The sign of a runtime zero is read without dividing by it, so --overflow-check
# raises no floating-point division claim.
def f(v: float) -> None:
    assert str(v) == "-0.0"


f(-0.0)
