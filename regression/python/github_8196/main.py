# Issue #8196: a method reaches a class attribute through self; a bare name
# resolves to the module-level binding, not the class attribute.
K: int = 5


class C:
    K: int = 2

    def m(self) -> int:
        return self.K + K


assert C().m() == 7
