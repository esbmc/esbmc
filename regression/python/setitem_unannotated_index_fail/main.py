class Box:
    def __init__(self):
        self.items = [1, 2, 3, 4]

    def __setitem__(self, i, v):
        self.items[i] = v

    def __getitem__(self, i: int) -> int:
        return self.items[i]


b = Box()
b[3] = 7
assert b[3] == 4
