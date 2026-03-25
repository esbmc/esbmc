from typing import Tuple

class MyClass:
    def __init__(self) -> None:
        self.value = 0

    def __setitem__(self, key:Tuple[int,int], value: int) -> None:
        i,j = key
        assert i == 4
        assert j == 5
        assert value == 12
        self.value = value


obj = MyClass()
obj[2, 3] = 11
