# The receiver must not land in x, or the result would be the instance (#8409).
class C:
    @staticmethod
    def ident(x: int) -> int:
        return x


n = nondet_int()
assert C().ident(n) == n
