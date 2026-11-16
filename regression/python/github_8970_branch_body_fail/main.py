# As github_8970_branch_body, but the int path asserts what is false.
def f():
    return "a"


def g():
    return 1


use = nondet_bool()
h = f if nondet_bool() else g
if use:
    r = h()
    assert r != 1
mark = 1
