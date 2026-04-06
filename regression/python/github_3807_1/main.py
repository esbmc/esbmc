# Test: for item in d.items() — tuple content is (key, value)
d: dict[str, int] = {"a": 1}
for item in d.items():
    k: str = item[0]
    v: int = item[1]
    assert k == "a"
    assert v == 1
