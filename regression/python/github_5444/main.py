def get1(d, k):
    return d[k]


key = "ab" if nondet_bool() else "c"
d = {key: 1}
assert get1(d, key) == 1


def get(d: dict, k: str) -> int:
    return d[k]


e = {"c": 2}
assert get(e, "c") == 2
