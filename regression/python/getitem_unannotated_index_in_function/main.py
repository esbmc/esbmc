class Box:
    def __init__(self):
        self.items = [1, 2, 3, 4]

    def __getitem__(self, i):
        return self.items[i]


def f() -> int:
    b = Box()
    return b[3]


assert f() == 4
