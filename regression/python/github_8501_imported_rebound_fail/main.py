import lib
from lib import f
lib.f(3)
assert not f("ab")
def f(x):
    return False
