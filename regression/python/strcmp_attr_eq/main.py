# An unannotated parameter is modelled as void*; comparing it against a string
# literal lowers to `strcmp(a, b) == 0` (converter_binop.cpp). Exercises the
# IREP2 equality2tc strcmp comparison (Part V Phase V.3).
class Box:
    def __init__(self):
        self.label = "hi"


def check(b) -> bool:
    return b.label == "hi"


assert check(Box()) == True
