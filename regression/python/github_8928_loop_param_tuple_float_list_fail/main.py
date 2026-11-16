# Unpacking for-loop over an unannotated parameter typed from its call site
# (#8928).
def total(xs):
    for (r, m) in xs:
        assert r[0] * m == 2.0


total([([1.0, 2.0], 3.0)])
