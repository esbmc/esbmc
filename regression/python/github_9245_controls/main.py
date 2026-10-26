from mod import smth
from modf import plain


class Local:
    def func(self, x=None):
        if x is None:
            return "a"
        return x


v = nondet_int()
assert Local().func(v) == v
assert plain(v) == v
assert smth.func(5) == 5
assert smth.func() == "a"
