def first_positive(p) -> list[int] | None:
    if p[0] > 0:
        return p
    return None


def last_positive(p) -> None | list[int]:
    if p[-1] > 0:
        return p
    return None


r = first_positive([1, 2])
assert r is not None
assert r[0] == 1 and len(r) == 2
assert first_positive([1]) == [1]
assert first_positive([0]) is None
assert last_positive([0, 3]) == [0, 3]
assert last_positive([3, 0]) is None
