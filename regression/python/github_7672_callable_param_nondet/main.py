# The SV-COMP shape from #7672: a nondet source passed through a
# `Callable`-annotated parameter. The assumption constrains the value the
# callee returns, so it can only hold at the caller if the call through the
# parameter carries that value back.
from typing import Callable


def nondet_int() -> int: ...


def source() -> int:
    v: int = nondet_int()
    __ESBMC_assume(v >= 10)
    return v


def call(g: Callable):
    return g()


def main() -> None:
    assert call(source) >= 10


main()
