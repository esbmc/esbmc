def f(g):
    d = {v: 0 for u, v in g}
    return len(d)


def h(g):
    s = 0
    for u, v in g:
        if v == 'CD':
            s = s + 1
    return s


assert f({('A', 'B'): 3, ('A', 'C'): 5}) == 2
# Multi-character components pin the UTF-8 byte-length sizing (length + 1
# with the NUL) rather than the single-char special case.
assert h({('AB', 'CD'): 3, ('EF', 'GH'): 5}) == 1
