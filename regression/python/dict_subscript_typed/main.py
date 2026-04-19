d: dict[int, float] = {1: 3.14}

x = d[1]

# ESBMC should know that x is a float
assert isinstance(x, float)

# And NOT an int
assert not isinstance(x, int)

# Float arithmetic must work
assert abs(x + 1.0 - 4.14) < 1e-6
