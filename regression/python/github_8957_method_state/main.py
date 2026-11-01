# The methods read the instance state: pick looks at self.c, g returns self.n.
class K:
    def __init__(self, c, n):
        self.c = c
        self.n = n

    def f(self):
        return "a"

    def g(self):
        return self.n

    def pick(self):
        if self.c:
            return self.f
        return self.g


c = nondet_bool()
n = nondet_int()
a = K(c, n)
r = a.pick()()
if c:
    assert r == "a"
else:
    assert r == n
