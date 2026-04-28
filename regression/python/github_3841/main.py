from collections import defaultdict

# Basic defaultdict with int factory - no missing key access
d: dict = defaultdict(int)
d["a"] = 1
d["b"] = 2
assert d["a"] == 1
assert d["b"] == 2
