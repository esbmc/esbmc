# Negative variant of forward_func_class_return: the same forward-referenced
# function returning a user-defined class, but with an assertion that does not
# hold. Confirms the fix preserves soundness (the previously-crashing path now
# yields a real verdict, and a false claim is correctly refuted).

class Widget:
    def __init__(self):
        self.x = 0.0


def f() -> float:
    w = make()
    return w.x


def make() -> Widget:
    return Widget()


assert f() == 99.0
