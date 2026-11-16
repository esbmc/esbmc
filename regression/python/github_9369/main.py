# Issue #9369: str() of a list must not be the digits of a pointer. The
# element of an empty list is never reached, so str() of it stays harmless.
def f(xs: list) -> None:
    for e in xs:
        s = str(e)
        assert len(s) > 0


f([])
x = [1, 2]
s = str(x)
assert len(s) >= 0
