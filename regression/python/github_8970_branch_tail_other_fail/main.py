# The rest of the module is copied into the branch that does not call h too,
# where this assertion is false.
def f():
    return "a"


def g():
    return 1


use = nondet_bool()
h = f if nondet_bool() else g
if use:
    r = h()
assert use
