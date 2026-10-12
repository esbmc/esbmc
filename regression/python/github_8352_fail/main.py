# The instance built through the variable is checked, not assumed (#8352).
class A:
    def m(self) -> int:
        return 1


cls = A
o = cls()
assert o.m() == 2
