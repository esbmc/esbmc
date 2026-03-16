from collections import defaultdict

# Missing-key read: accessing a key never assigned should auto-insert
# the factory-produced value (0 for int) and return it.
d: dict = defaultdict(int)
v: int = d["missing"]
assert v == 0
assert d["missing"] == 0
