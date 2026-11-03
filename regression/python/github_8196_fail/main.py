# Issue #8196: a class body is not an enclosing scope for its methods, so a
# bare class attribute name in a method is undefined (CPython: NameError).
class C:
    K: int = 2

    def m(self):
        return K


assert C().m() == 2
