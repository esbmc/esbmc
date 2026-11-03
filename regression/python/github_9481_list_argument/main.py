class B:
    def __init__(self, s):
        self.s = s


b = B([])
b.s.append(1)
b.s.append(1)
assert len(b.s) == 2
