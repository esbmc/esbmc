# Issue #8226: the nested names hold the element's values.
def advance(pairs):
    for ((x1, y1), (x2, y2)) in pairs:
        assert y2 == 3


advance([((1, 2), (3, 4))])
