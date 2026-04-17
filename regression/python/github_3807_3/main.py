# Regression: for k, v in d.items() (tuple unpacking) still works after fix
d: dict[str, int] = {"a": 1, "b": 2}
count: int = 0
for k, v in d.items():
    count = count + 1
assert count == 2
