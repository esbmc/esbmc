# g takes a defaulted parameter, so a call h(1) is not rewritten to name it.
def f(x):
    return "a"


def g(x, y=0):
    return 1


def call(h):
    return h(1)


c = nondet_bool()
r = call(f if c else g)
