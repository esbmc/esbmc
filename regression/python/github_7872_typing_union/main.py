# esbmc/esbmc#7872 in its typing.Union spelling: both forms reach the same
# widening, so both need a gate.
from typing import Union, List


def foo(y: List[int]) -> Union[int, List[int]]:
    return y


M = [1, 2, 3]

assert foo(M) == M
