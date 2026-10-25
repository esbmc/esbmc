class A:

    def name(self) -> str:
        return self.__class__.__name__


class B:
    pass


a = A()
f = a.name
a.__class__ = B
assert f() == "B"
