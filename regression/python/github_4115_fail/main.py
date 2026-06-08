class Leaf:
    def __init__(self, v):
        self.value = v


class Holder:
    def __init__(self):
        self.leaf = Leaf(1)


h = Holder()
h.leaf.value = 99
assert h.leaf.value == 100
