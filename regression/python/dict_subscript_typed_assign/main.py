# Typed dictionary: int -> float
d: dict[int, float] = {1: 1.0}

# Correct assignment: value is a float
d[2] = 3.5

# ESBMC should know this is valid
assert isinstance(d[2], float)
assert d[2] == 3.5

# Ensure the previous entry is preserved
assert d[1] == 1.0

# Ensure the dict did not accidentally change type
assert not isinstance(d[2], int)

