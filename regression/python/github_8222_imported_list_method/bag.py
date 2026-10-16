class Bag:

    def __init__(self) -> None:
        self.items: list[int] = [1, 2]

    def __len__(self) -> int:
        return len(self.items)

    def append(self, v: int) -> None:
        self.items.append(v * 2)

    def extend(self, v: int) -> None:
        self.items.append(v)
        self.items.append(v)

    def clear(self) -> None:
        self.items = [0]
