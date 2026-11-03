# Issue #8200: a method default is evaluated in the class body, so a name the
# body bound before the method is the class attribute, otherwise the global.
G: int = 7


class C:
    K: int = 2

    def f(self, x=K, *, y=K, z=G):
        return x + y + z


assert C().f() == 11
