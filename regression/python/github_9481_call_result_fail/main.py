def make_set():
    return {1, 2}


class B:
    def __init__(self):
        self.s = make_set()


b = B()
b.s.add(2)
b.s.add(3)
assert len(b.s) == 4
