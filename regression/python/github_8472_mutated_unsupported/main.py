# The callee shifts the elements before reading one, so the positions the
# call passes no longer hold: refused rather than misread (#8472).
def second_first(d):
    d.insert(0, 5)
    return d[1][0]


assert second_first([[1, 2], 3]) == 1
