# github #9629: repr() calls __repr__, and str() falls back to it.
class R:
    def __repr__(self):
        return "<R>"


class S:
    def __str__(self):
        return "<S>"


class A:
    def __repr__(self):
        return "<A>"


class B(A):
    def __str__(self):
        return "B"


r = R()
assert repr(r) == "<R>"
assert str(r) == "<R>"

s = S()
assert repr(s) != "<S>"
assert repr(s).startswith("<__main__.S object at 0x")

b = B()
assert str(b) == "B"
assert repr(b) == "<A>"
