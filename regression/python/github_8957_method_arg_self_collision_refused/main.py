# run reads self and has a parameter named like the instance, so lifting it would capture the name.
class K:
    def __init__(self):
        self.n = 1

    def f(self):
        return "s"

    def g(self):
        return 1

    def call(self, x):
        return x()

    def run(self, a, b):
        k = self.n
        return a(b)


a = K()
c = nondet_bool()
r = a.run(a.call, a.f if c else a.g)
