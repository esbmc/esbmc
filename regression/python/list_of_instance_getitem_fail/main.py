class C:
    def __init__(self) -> None:
        self.items: list[int] = [1, 2]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> int:
        return self.items[i]

a = C()
b = list(a)
assert b == [1, 3]
