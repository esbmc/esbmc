class Box:
    def __init__(self):
        self.items = [1, 2, 3, 4]

    def __getitem__(self, i: int) -> int:
        return self.items[i]

    def __setitem__(self, i: int, v: int) -> None:
        self.items[i] = v


boxes = [Box(), Box()]
boxes[1][3] = 9
assert boxes[1][3] == 9
assert boxes[0][3] == 9
