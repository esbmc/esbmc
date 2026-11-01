# pick returns the result of a call on one path, so it does not return a bound method there.
class K:
    def f(self):
        return "a"

    def g(self):
        return 1

    def pick(self, c):
        if c:
            return self.f
        return self.g()


a = K()
c = nondet_bool()
r = a.pick(c)()
