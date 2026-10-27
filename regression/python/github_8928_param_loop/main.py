# GitHub #8928: tuples of a float list and a float iterated through a parameter
def total(xs):
    for (r, m) in xs:
        assert r[0] * m == 3.0


total([([1.0, 2.0], 3.0)])
