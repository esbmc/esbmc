# The non-callable arm raises only when it is called
def f():
    return 1


def guarded(a):
    try:
        return a()
    except TypeError:
        return 0


def ignored(a):
    return 5


r = guarded(f if nondet_bool() else 3)
assert r == 0 or r == 1
assert ignored(f if nondet_bool() else 3) == 5
