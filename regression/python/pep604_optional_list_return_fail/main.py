def first_positive(p) -> list[int] | None:
    if p[0] > 0:
        return p
    return None


r = first_positive([0])
assert r[0] == 0
