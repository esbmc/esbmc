# Issue #8226: a for-loop target with nested tuples binds every name when the
# loop iterates a parameter.
def advance(pairs):
    total = 0
    for ((x1, y1), (x2, y2)) in pairs:
        assert x1 - x2 == -2
        total += y1 + y2
    for (k, (a, b)) in [(1, (2, 3))]:
        total += k * a * b
    assert total == 6 + 14 + 6


advance([((1, 2), (3, 4)), ((5, 6), (7, 8))])
