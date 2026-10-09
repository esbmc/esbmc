# A method default naming a class attribute bound after the method reads the
# global: the default is evaluated where the def runs (#8200).
K: int = 5


class C:
    def f(self, x=K):
        return x

    K: int = 2


assert C().f() == 5
