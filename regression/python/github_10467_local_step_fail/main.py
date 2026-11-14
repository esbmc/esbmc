def every(l: list[int]) -> list[int]:
    s = 3
    return l[1::s]


r = every([10, 20, 30, 40, 50, 60, 70])
assert len(r) == 7
