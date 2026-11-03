i = 0


def w() -> str:
    global i
    i += 1
    return "x" if i < 4 else "y"


while w() == "x":
    pass
assert i == 4
