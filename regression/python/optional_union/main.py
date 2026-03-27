from typing import Union


def foo(x: Union[int, None]) -> None:
    if x is None:
        assert True
    else:
        assert isinstance(x, int)


foo(None)
foo(5)
