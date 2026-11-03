class Schema:

    def __init__(self, n: int):
        self.n = n

    def describe(self) -> str:
        return self.__class__.__name__ + str(10 // self.n)


s = Schema(1)
assert s.describe() == "Schema10"
assert type(s).__name__ == "Schema"
