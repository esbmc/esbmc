class Box:
    def __init__(self):
        self.items = [1, 2, 3, 4]

    def __getitem__(self, i: int) -> int:
        return self.items[i]


boxes = [Box(), Box()]
assert boxes[1][3] == 4
