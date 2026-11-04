# A second instance is bound as well, so a is not the only receiver.
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
a = K()
c = nondet_bool()
r = a.pick(c)()
