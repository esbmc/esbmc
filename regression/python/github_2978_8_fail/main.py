from typing import Literal, Union


def foo(s: Union[Literal["foo"], Literal["bar"]]) -> int:
    return 42


assert foo("foo") == 41
