# esbmc/esbmc#7876 in its typing.Union spelling. This one already reached the
# parameter as an opaque pointer, so only the argument check rejected it.
from typing import Union, List


def foo(y: Union[int, List[int]]) -> List[int]:
    return y


M = [1, 2, 3]

assert foo(M) == M
