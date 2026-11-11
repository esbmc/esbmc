# #10231: pick() has no top-level return, so its call was typed NoneType and
# `r = f()` was refused with Function "f" not found.
def g():
    return "a"


def pick(flag: bool):
    if flag:
        return g


f = pick(True)
r = f()
assert r == "b"
