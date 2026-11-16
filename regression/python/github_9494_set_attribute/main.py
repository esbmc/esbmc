class B:
    def __init__(self):
        self.s = set()


b = B()
b.s.add(1)
b.s.add(1)
assert len(b.s) == 1
