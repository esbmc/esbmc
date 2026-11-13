import random

from pkg.fwd import func1

a = func1()
assert isinstance(a, str)
assert a == "Hello"
n = random.randint(0, 9)
b = func1(n)
assert isinstance(b, str)
