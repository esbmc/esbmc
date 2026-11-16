
def first_up(p):
    for i in range(len(p) - 1):
        if p[i] < p[i + 1]:
            q = list(p)
            q[i], q[i + 1] = p[i + 1], p[i]
            return q


r = first_up([3, 1, 2])
assert r is not None and len(r) == 3
assert r[0] == 3 and r[1] == 2 and r[2] == 1
assert first_up([2, 1]) is None
