d: dict[int, str] = {1: "not-a-float"}  # Wrong type, ESBMC must detect mismatch

x = d[1]

# This assertion must fail because x is *not* a float
assert isinstance(x, float)

