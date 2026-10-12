# A module-level call before the alias is bound is not resolved to the
# class: CPython raises NameError there (#8352).
class A:
    def m(self) -> int:
        return 1


o = cls()
cls = A
assert o.m() == 1
