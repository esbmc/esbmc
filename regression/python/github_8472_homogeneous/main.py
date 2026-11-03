# Control: a list parameter whose elements are all lists keeps its single
# element type (#8472).
def t(d):
    a = d[0]
    return a[1]


def is_list(d):
    return isinstance(d[1], list)


assert t([[1, 2], [3]]) == 2
assert is_list([[1, 2], [3]])
