class MyClass:
    def __init__(self) -> None:
        self.value = 0

    def __getitem__(self, key: int) -> int:
        assert key == 2
        return self.value

    def __setitem__(self, key: int, value: int) -> None:
        assert key == 2
        assert value == 11
        self.value = value


obj = MyClass()
obj[2] = 11
v:int = obj[2]
assert v == 11
