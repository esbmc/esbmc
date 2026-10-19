# A bare `dict` parameter takes K, V from its call sites (#5444).
def get(d: dict, k: str) -> int:
    return d[k]


def first(d: dict, k: str) -> float:
    return d[k]


a = {"c": 2}
b: dict[str, int] = {"c": 3}
c: dict = {"c": 0.5}
assert get(a, "c") == 2
assert get(b, "c") == 3
assert first(c, "c") == 0.5
