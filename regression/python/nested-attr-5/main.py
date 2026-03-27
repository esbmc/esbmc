# Test case 5: Nested attribute in conditional statements
class Counter:
    def __init__(self) -> None:
        self.count: int = 0

    def increment(self) -> int:
        self.count += 1
        return self.count

    def get_count(self) -> int:
        return self.count

class Manager:
    def __init__(self) -> None:
        self.counter: Counter = Counter()

    def process(self) -> int:
        # Use nested attribute in conditional
        if self.counter.get_count() < 5:
            result = self.counter.increment()
            return result
        return self.counter.get_count()

manager = Manager()
result: int = manager.process()
assert result == 1

