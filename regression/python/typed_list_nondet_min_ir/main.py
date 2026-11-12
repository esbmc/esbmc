xs: list[int] = nondet_list()
if len(xs) > 0:
    assert xs[0] != -9223372036854775807 - 1
