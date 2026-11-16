def ident(x):
    return x


class C:
    @staticmethod
    def twice(x):
        return x


assert ident(3) == 3
assert C.twice(4) == 4
