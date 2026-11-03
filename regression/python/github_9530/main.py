def make():
    return [1, 2]


class B:
    def __init__(self):
        self.s = make()


b = B()
if len(b.s) == 3:
    b.s.foo(1)
assert len(b.s) == 2
