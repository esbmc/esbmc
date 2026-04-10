sign = lambda x: "pos" if x > 0 else "neg" if x < 0 else "zero"
assert sign(3) == "pos"
assert sign(-2) == "neg"
assert sign(0) == "zero"

abs_val = lambda x: x if x >= 0 else -x
assert abs_val(-7) == 7
