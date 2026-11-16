# As github_8970_branch_elif_else, but the else path asserts what is false.
def f():
    return "a"


def g():
    return 1


first = nondet_bool()
second = nondet_bool()
h = f if nondet_bool() else g
if first:
    tag = 1
elif second:
    r = h()
    tag = 2
else:
    r = h()
    assert r != "a"
    tag = 3
