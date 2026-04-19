# Test: for item in d.items() with multiple entries
d: dict[str, int] = {"x": 10, "y": 20}
count: int = 0
for item in d.items():
    assert len(item) == 2
    count = count + 1
assert count == 2
