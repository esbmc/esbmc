class Box:
    def __init__(self):
        self.items = [1, 2, 3, 4]

    def __getitem__(self, i):
        return self.items[i]


b = Box()
assert b.__getitem__(3) == 4
