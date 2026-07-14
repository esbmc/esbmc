# NotEq variant of the void*-vs-string strcmp path: `b.label != "hi"` lowers to
# `strcmp(a, b) != 0` (IREP2 notequal2tc). The label equals "hi", so the
# comparison is False and the assertion is violated (Part V Phase V.3).
class Box:
    def __init__(self):
        self.label = "hi"


def check(b) -> bool:
    return b.label != "hi"


assert check(Box()) == True
