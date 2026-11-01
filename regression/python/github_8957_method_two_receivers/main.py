# Two instances each return their own bound method, and each call goes through its own receiver.
class K:
    def __init__(self, n):
        self.n = n

    def f(self):
        return "a"

    def g(self):
        return self.n

    def pick(self, c):
        if c:
            return self.f
        return self.g


a = K(1)
b = K(2)
c = nondet_bool()
r = a.pick(c)()
s = b.pick(c)()
if c:
    assert r == "a" and s == "a"
else:
    assert r == 1 and s == 2
