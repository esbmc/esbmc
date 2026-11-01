# The call is a.pick(c)(), whose callee is a call.
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
r = a.pick(c)()
assert (r == "a") == c
