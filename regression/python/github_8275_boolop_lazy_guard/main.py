calls: int = 0


def g(v: int) -> int:
    global calls
    calls += 1
    return v


def at(a: list[int], j: int) -> bool:
    return j < len(a) and a[j] == 1


x: int = 0
ok: bool = x != 0 and 10 // x > 1
assert not ok
ok = x == 0 or 10 // x > 1
assert ok
ok = x == 1 and 5 // g(x) > 0
assert calls == 0
y: int = x or 7 // (x + 1)
assert y == 7

xs: list[int] = [1, 2, 3]
i: int = 3
assert not (i < len(xs) and xs[i] > 0)
assert not at([], 2)
assert at([1], 0)

d: dict[str, int] = {"a": 1}
k: str = "b"
assert not (k in d and d[k] > 0)
