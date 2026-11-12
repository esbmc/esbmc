# Issue #10287: an augmented assignment whose operator raises TypeError.
def f() -> None:
    a = "6"
    a -= 1


def g() -> None:
    a = "6"
    a /= 2


def h() -> None:
    a = 6
    a -= "1"


caught = 0
try:
    f()
except TypeError:
    caught += 1
try:
    g()
except TypeError:
    caught += 1
try:
    h()
except TypeError:
    caught += 1
assert caught == 3
