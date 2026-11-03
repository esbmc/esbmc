# As github_9324_index_expr, but the odd path asserts what is false.
def f():
    return 1


def g():
    return "a"


fs = [f, g]
i = nondet_int()
r = fs[i % 2]()
if i % 2 == 1:
    assert r == 1
