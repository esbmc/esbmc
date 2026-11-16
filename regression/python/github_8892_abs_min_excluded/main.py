x: int = nondet_int()
if x != -9223372036854775807 - 1:
    assert abs(x) >= 0
