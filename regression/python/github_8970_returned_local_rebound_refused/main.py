# h is rebound to a value that is no function when c is None, which run never gets.
def f():
    return "a"


def g():
    return 1


def run(c):
    h = f if c else g
    if c is None:
        h = 5
    return h()


n = nondet_bool()
r = run(n)
assert (r == "a") == n
