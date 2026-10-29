# The method value is also stored, which the lift does not follow.
class K:
    def f(self):
        return "s"

    def g(self):
        return 1

    def call(self, x):
        return x()


a = K()
c = nondet_bool()
h = a.f
r = a.call(a.f if c else a.g)
h()
