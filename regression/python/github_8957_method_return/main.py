# pick returns a bound method of a, and the result is called.
class K:
    def f(self):
        return "a"

    def g(self):
        return 1

    def pick(self, c):
        if c:
            return self.f
        return self.g


a = K()
c = nondet_bool()
h = a.pick(c)
r = h()
if c:
    assert r == "a"
else:
    assert r == 1
