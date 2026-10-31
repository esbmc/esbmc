class B:
    def __init__(self):
        self.s = []


b = B()
b.s.append(1)
b.s.append(2)
assert b.s.pop() == 2
assert b.s.index(1) == 0
assert len(b.s) == 1
