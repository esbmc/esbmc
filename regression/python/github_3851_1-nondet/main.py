# Global declared after class; method branches on a nondet int global.
# If counter >= 0, pre() returns True; otherwise False.
# The assertion must hold under all nondet values satisfying the assumption.

class A:
    def pre(self) -> bool:
        return counter >= 0


counter: int = nondet_int()
__ESBMC_assume(counter >= 0)

a = A()
assert a.pre()
