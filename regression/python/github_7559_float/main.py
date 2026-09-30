# !r, !a and = on a float render its repr, which is its str() (#7559); a
# constant folds to CPython's shortest round-trip digits and layout.
def f(v: float, z: float) -> None:
    assert f"{v!r}" == "2.5" and f"{v!a}" == "2.5"
    assert f"{v=}" == "v=2.5"
    assert repr(z) == "-0.0" and ascii(z) == "-0.0"


f(2.5, -0.0)
assert f"{10000000000000002.0!r}" == "1.0000000000000002e+16"
assert str(5.960464477539063e-08) == "5.960464477539063e-08"
assert repr(1e16) == "1e+16" and str(1e22) == "1e+22"
