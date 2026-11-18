class Box:
    def __init__(self, **kw):
        self._count = len(kw)


b = Box(a=1, b=2)
assert b._count == 3
