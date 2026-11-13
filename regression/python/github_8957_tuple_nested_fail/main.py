class K:
    def __init__(self, n):
        self.n = n

    def m1(self):
        return self.n

    def m2(self):
        return 2.5

    def m3(self):
        return "a"


class L:
    pass


n = nondet_int()
k, l = K(n), L()
swap = nondet_bool()
if swap:
    c, (d, e) = k.m3, (k.m2, k.m1)
else:
    c, (d, e) = k.m1, (k.m2, k.m3)
f = c()
g = d()
h = e()
assert g == 2.5
if swap:
    assert f == "a"
    assert h == "a"
else:
    assert f == n
    assert h == "a"
