# The method reads self and has a parameter named like the instance, so the lift would capture it.
class K:
    def f(self):
        return "s"

    def g(self):
        return 1

    def run(self, a, b):
        return a(b) if self.g() else None

    def call(self, x):
        return x()


a = K()
c = nondet_bool()
r = a.run(a.call, a.f if c else a.g)
