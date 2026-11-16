# As github_8957_method_return, but the int path asserts what is false.
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
h = a.pick(c)
r = h()
if not c:
    assert r == "a"
