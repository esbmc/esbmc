def test() -> None:
    # Tuple keys whose elements are strings: the key struct holds char arrays,
    # not pointers, so its byte size must be measured exactly (regression for an
    # out-of-bounds copy that treated each char[] component as a pointer).
    d = {("x", "y"): 1, ("x", "z"): 2}
    assert d[("x", "y")] == 1
    assert d[("x", "z")] == 2


test()
