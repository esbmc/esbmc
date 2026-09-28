# The destructured components are really read, not assumed.
def f(d):
    for edge in d:
        u, v = edge
        assert u > v


f({(1, 2): 10, (3, 4): 20})
