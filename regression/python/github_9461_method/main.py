class B:
    def __init__(self):
        self.s = set()

    def put(self, p):
        self.s.add(p)


b = B()
b.put(1)
b.put(1)
b.put(2)
assert len(b.s) == 2
assert 1 in b.s
assert 3 not in b.s
