class Counter:

    def __init__(self) -> None:
        self.data: dict[int, int]
        self.data = {}

    def __getitem__(self, key: int) -> int:
        try:
            return self.data[key]
        except KeyError:
            return 0

    def __setitem__(self, key: int, value: int) -> None:
        self.data[key] = value


c = Counter()
assert c[2] == 1
