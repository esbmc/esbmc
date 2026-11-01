# The call passes fewer arguments than the method takes, so it is not lifted.
class K:
    def f(self):
        return "s"

    def g(self):
        return 1

    def call(self, x, y=0):
        return x()


a = K()
c = nondet_bool()
r = a.call(a.f if c else a.g)
