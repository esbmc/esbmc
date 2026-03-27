# Test case 3: Nested attribute with different return types
class StringWrapper:
    def get_str(self) -> str:
        return "hello"

class IntWrapper:
    def get_int(self) -> int:
        return 100

class Container:
    def __init__(self) -> None:
        self.str_wrapper: StringWrapper = StringWrapper()
        self.int_wrapper: IntWrapper = IntWrapper()

    def get_string(self) -> str:
        # Test nested attribute returning string
        result = self.str_wrapper.get_str()
        return result

    def get_integer(self) -> int:
        # Test nested attribute returning integer
        result = self.int_wrapper.get_int()
        return result

container = Container()
s: str = container.get_string()
i: int = container.get_integer()
assert s == "hello"
assert i == 100

