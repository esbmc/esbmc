# #9377: an empty str operand of and/or is false.
def g() -> str:
    return ""


def h() -> str:
    return "y"


s = ""
t = ""
assert (s or "x") == "x"
assert (s and "b") == ""
assert (s or t or "z") == "z"
assert (g() or h()) == "y"
u = "a"
assert (u or "x") == "a"
assert (u and "b") == "b"
