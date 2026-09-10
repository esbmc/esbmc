# `source` is only assumed >= 10, so a stronger bound must stay refutable: the
# counterpart guards against the call through the parameter being folded to a
# constant instead of carrying the nondet value (#7672).
from typing import Callable


def nondet_int() -> int: ...


def source() -> int:
    v: int = nondet_int()
    __ESBMC_assume(v >= 10)
    return v


def call(g: Callable):
    return g()


def main() -> None:
    assert call(source) >= 11


main()
