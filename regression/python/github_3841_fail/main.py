from collections import defaultdict

# Should fail: assertion is false
d: dict = defaultdict(int)
d["a"] = 5
assert d["a"] == 99  # Wrong: value is 5, not 99
