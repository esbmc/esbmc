# An alias bound inside a function is that function's local, so a
# module-level call through it raises NameError in CPython (#8352).
class A:
    def m(self) -> int:
        return 1


def setup() -> None:
    cls = A


setup()
o = cls()
assert o.m() == 1
