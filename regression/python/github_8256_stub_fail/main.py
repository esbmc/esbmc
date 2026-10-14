# Issue #8256: the stub's method is called, so only its ZeroDivisionError is
# reported, not an unsupported-function property.
class np:

    class random:

        @staticmethod
        def randint(a0: int) -> int:
            v: int = nondet_int()
            return v


def f(n: int) -> int:
    k: int = np.random.randint(n)
    return n // k


f(nondet_int())
