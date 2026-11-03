x = 2.5
if nondet_bool():
    x = 1.5
assert f"{x:.6f}" == "1.500000"
