def dict_sum(g):
    s = 0
    for k, w in g.items():
        s += w
    return s


def list_sum(g):
    s = 0
    for x in g:
        s += x
    return s


assert dict_sum({'a': 3, 'b': 7}) == 10
assert list_sum([1, 2, 3]) == 6
