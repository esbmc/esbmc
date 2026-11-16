# The returned method is kept in a list.
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
fs = [a.pick(c)]
r = fs[0]()
