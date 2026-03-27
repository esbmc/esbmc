from typing import Tuple


class MyClass:

    def __init__(self) -> None:
        self.value = 0

    def __getitem__(self, key: Tuple[int, int]) -> int:
        i, j = key
        assert i == 2
        assert j == 3
        return self.value

    def __setitem__(self, key: Tuple[int, int], value: int) -> None:
        i, j = key
        assert i == 2
        assert j == 3
        assert value == 11
        self.value = value


obj = MyClass()
obj[2, 3] = 11
v: int = obj[2, 3]
assert v == 11
