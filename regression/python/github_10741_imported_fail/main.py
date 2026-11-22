# call(f()) calls a str, which raises TypeError when flag holds.
from helper import call


def f():
    return "s"


def g():
    return 7


r = call(f() if nondet_bool() else g)
assert r == 7
