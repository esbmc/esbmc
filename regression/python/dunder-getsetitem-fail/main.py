class MyClass:

    def __init__(self) -> None:
        self.value = 0

    def __setitem__(self, key: int, value: int) -> None:
        assert key == 3  #key == 2
        assert value == 12  #value == 11
        self.value = value


obj = MyClass()
obj[2] = 11
