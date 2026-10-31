class B:
    def __init__(self, s):
        self.s = s


b = B(s={1})
b.s.add(1)
b.s.add(2)
assert len(b.s) == 2
