# Issue #9156: a method returning a module-level name holding a str or an int.
c = nondet_bool()
if c:
    r = "a"
else:
    r = 1


class C:

    def m(self):
        return r


z = C().m()
assert isinstance(z, str)
