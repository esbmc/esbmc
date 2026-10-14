# Counterpart to github_7872_typing_union.
from typing import Union, List


def foo(y: List[int]) -> Union[int, List[int]]:
    return y


M = [1, 2, 3]

assert foo(M) != M
