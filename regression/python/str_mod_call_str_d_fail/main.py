# CPython raises TypeError for %d on a str; the operand stays refused.
def s() -> str:
    return "x"


t = "id %d" % s()
