# The method is read with getattr.
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
r = getattr(a, "pick")(c)()
