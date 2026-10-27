# Unpacking for-loop over an unannotated parameter typed from its call site
# (#8928).
def total(xs):
    for (r, m) in xs:
        assert r[0] * m == 3.0


def scaled(xs) -> float:
    s = 0.0
    for r, m in xs:
        s += r[1] * m
    return s


total([([1.0, 2.0], 3.0)])
assert scaled([([1.0, 2.0], 3.0), ([0.0, 0.5], 4.0)]) == 8.0
