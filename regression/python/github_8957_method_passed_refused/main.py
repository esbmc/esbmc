# a is passed to a function, which may rebind or store it.
class K:
    def f(self):
        return "a"

    def g(self):
        return 1

    def pick(self, c):
        if c:
            return self.f
        return self.g


def use(x):
    return x.pick(True)


a = K()
c = nondet_bool()
r = a.pick(c)()
use(a)
