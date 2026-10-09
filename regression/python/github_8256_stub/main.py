# Issue #8256: a hand-written numpy.random stub resolves to its own method.
class np:

    class random:

        @staticmethod
        def randint(a0: int) -> int:
            v: int = nondet_int()
            __ESBMC_assume(v != 0)
            return v


def f(n: int) -> int:
    k: int = np.random.randint(n)
    return n // k


f(nondet_int())
