# Test case 11: Mixed nested attributes with different types
class StringProcessor:
    def process(self, s: str) -> str:
        return "HELLO" if s == "hello" else s

class IntProcessor:
    def process(self, i: int) -> int:
        return i * 2

class Processor:
    def __init__(self) -> None:
        self.str_proc: StringProcessor = StringProcessor()
        self.int_proc: IntProcessor = IntProcessor()

class Service:
    def __init__(self) -> None:
        self.processor: Processor = Processor()

    def handle_string(self, s: str) -> str:
        # Nested attribute returning string
        result = self.processor.str_proc.process(s)
        return result

    def handle_int(self, i: int) -> int:
        # Nested attribute returning int
        result = self.processor.int_proc.process(i)
        return result

service = Service()
s_result: str = service.handle_string("hello")
i_result: int = service.handle_int(5)
assert s_result == "HELLO"
assert i_result == 10
