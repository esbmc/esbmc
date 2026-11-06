class C:
    def __init__(self) -> None:
        self.items: list[int] = [1, 2]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> int:
        return self.items[i]


def f(c: C) -> list[int]:
    return list(c)

assert f(C()) == [1, 2]
