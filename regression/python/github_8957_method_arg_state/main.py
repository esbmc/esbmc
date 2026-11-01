# The passed methods read the instance state, and the receiver shares its name with a parameter.
class K:
    def __init__(self, n):
        self.n = n

    def f(self):
        return "s"

    def g(self) -> int:
        return self.n

    def call(self, a):
        return a()

    def twice(self, a, b):
        return a(b)


n = nondet_int()
a = K(n)
c = nondet_bool()
r = a.twice(a.call, a.f if c else a.g)
if c:
    assert r == "s"
else:
    assert r == n
