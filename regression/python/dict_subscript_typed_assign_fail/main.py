# Typed dictionary: int -> float
d: dict[int, float] = {1: 1.0}

# Wrong assignment: assigning a string where a float is expected
d[2] = "wrong-type"

x = d[2]

# ESBMC should detect mismatch and fail
assert isinstance(x, float)   # This must fail
