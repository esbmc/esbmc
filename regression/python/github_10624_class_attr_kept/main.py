class Box:
    made = []

    def __init__(self, v):
        self.v = v
        self.made.append(v)


b = Box(1.5)
a = Box(2)
assert len(a.made) == 2
