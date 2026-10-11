# A class called through a variable constructs that class (#8352).
class A:
    def m(self) -> int:
        return 1


cls = A
o = cls()
assert o.m() == 1
