def get(d: dict[str, int], k: str) -> int:
    return d[k]


def size(s: str) -> int:
    return len(s)


k = "ab"
if nondet_bool():
    k = "c"
d = {k: 3}
assert get(d, k) == 3

s = "c"
assert size(s) == 2
s = "ab"
assert size(s) == 2
