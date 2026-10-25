# GitHub #9148: int has no upper(), so the int path raises AttributeError.
if nondet_bool():
    r = "a"
else:
    r = 1
s = r.upper()
