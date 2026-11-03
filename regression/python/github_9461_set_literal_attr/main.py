class B:
    def __init__(self):
        self.s = {1}


b = B()
b.s.add(1)
b.s.add(2)
assert len(b.s) == 2
