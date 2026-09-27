# Counterpart to github_7876_typing_union.
from typing import Union, List


def foo(y: Union[int, List[int]]) -> List[int]:
    return y


M = [1, 2, 3]

assert foo(M) != M
