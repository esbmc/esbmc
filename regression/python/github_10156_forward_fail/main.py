# A defines __lt__, so Python calls it whatever its annotation says, and
# never B.__gt__ (#10156).
class C:
    pass


class A:
    def __lt__(self, o: C) -> bool:
        return True


class B:
    def __gt__(self, o: A) -> bool:
        return False


a = A()
b = B()
assert not (a < b)
