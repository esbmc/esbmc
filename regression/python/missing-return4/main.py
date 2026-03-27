class TestClass:

    def __init__(self, value: int):
        self.value = value
        # No return needed in constructor

    def get_value(self) -> int:
        return self.value


obj = TestClass(42)
assert obj.get_value() == 42
