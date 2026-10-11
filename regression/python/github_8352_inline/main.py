# A class called through a variable, used inline, constructs that class (#8352).
class A:
    def m(self) -> int:
        return 1


cls = A
assert cls().m() == 1
