# The MRO of D is D, L, R, A, so D.m is R's override, not A.m, which a
# depth-first search through L reaches first (#7546).
class A:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class L(A):
    pass


class R(A):
    @staticmethod
    def m(a: int, b: int) -> int:
        return a + b


class D(L, R):
    pass


# Read at module scope: a use of D does not rebind it.
assert D.m(10, 4) == 14
