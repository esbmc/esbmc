# Box holds an untyped h but has no g, so b.g() still raises AttributeError
# (#9255).
def f():
    return 1


class Box:
    def __init__(self, h):
        self.h = h


b = Box(f)
assert b.g() == 1
