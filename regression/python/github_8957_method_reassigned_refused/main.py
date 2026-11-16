# self.f is reassigned, so the bound method is no longer the class method.
class K:
    def f(self):
        return "a"

    def g(self):
        return 1

    def pick(self, c):
        if c:
            return self.f
        return self.g

    def rebind(self):
        self.f = self.g


a = K()
a.rebind()
c = nondet_bool()
r = a.pick(c)()
