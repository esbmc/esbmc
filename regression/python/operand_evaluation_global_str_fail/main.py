s = "a"


def bump() -> str:
    global s
    s = "b"
    return "c"


r = s + bump()
assert r == "bc"
