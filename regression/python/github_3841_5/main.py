from collections import defaultdict

# Missing-key read: accessing a key never assigned should auto-insert
# the factory-produced value (0 for int) and return it.
d: dict = defaultdict(int)
v: int = d["missing"]
assert v == 0

# Direct assert on a fresh key (no prior assignment) must also auto-insert.
assert d["another"] == 0
