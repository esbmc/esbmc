d1: dict[int, int] = {1: 2, 3: 4}
assert d1[1] == 2
assert d1[3] == 4

d2: dict[str, int] = {"hello": 42, "world": 100}
assert d2["hello"] == 42
assert d2["world"] == 100

d3: dict[int, str] = {1: "one", 2: "two"}
assert d3[1] == "one"
assert d3[2] == "two"

d4: dict[str, str] = {"key": "value"}
assert d4["key"] == "value"

d5: dict[int, bool] = {0: False, 1: True}
assert d5[0] == False
assert d5[1] == True

d6: dict[int, float] = {1: 3.14, 2: 2.71}
assert d6[1] == 3.14
assert d6[2] == 2.71

d: dict[int, float] = {1: 3.14}
x = d[1]
assert isinstance(x, float)

