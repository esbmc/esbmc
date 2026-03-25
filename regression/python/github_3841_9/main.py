from collections import defaultdict

# defaultdict(None) should behave like a plain dict (no auto-insertion).
# Explicitly assigned keys must still work correctly.
d: dict = defaultdict(None)
d["a"] = 42
assert d["a"] == 42
