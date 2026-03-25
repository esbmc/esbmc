from collections import defaultdict

# Explicit read of defaultdict value (missing key auto-insert)
d: dict = defaultdict(int)
d["x"] = 42
v: int = d["x"]
assert v == 42
