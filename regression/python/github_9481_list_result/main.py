def make():
    return [1, 2]


class B:
    def __init__(self):
        self.s = make()


b = B()
b.s.append(2)
assert len(b.s) == 3
