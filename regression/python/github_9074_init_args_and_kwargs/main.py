class Box:
    def __init__(self, first, *rest, **kw):
        self.total = first + len(rest) + len(kw)


b = Box(10, 20, 30, x=1)
assert b.total == 13
