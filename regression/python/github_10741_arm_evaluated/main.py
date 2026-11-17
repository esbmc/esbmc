# CPython evaluates k() before calling its value, so ValueError is raised,
# not TypeError.
def call(a):
    return a()


def k():
    raise ValueError("k")


def g():
    return 7


flag = nondet_bool()
try:
    r = call(k() if flag else g)
    assert not flag and r == 7
except ValueError:
    assert flag
