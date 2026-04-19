# Test: for item in d.items() on empty dict — loop body never executes
d: dict[str, int] = {}
count: int = 0
for item in d.items():
    count = count + 1
assert count == 0
