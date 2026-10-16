# A function stored on an attribute through an unannotated parameter is not
# a missing attribute: refuse the call rather than raise AttributeError (#9255).
def f():
    return 1


class Box:
    def __init__(self, h):
        self.h = h


b = Box(f)
assert b.h() == 1
