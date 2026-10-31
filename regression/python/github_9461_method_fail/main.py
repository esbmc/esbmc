class B:
    def __init__(self):
        self.s = set()

    def put(self, p):
        self.s.add(p)


b = B()
b.put(1)
b.put(1)
assert len(b.s) == 2
