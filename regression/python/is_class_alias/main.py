from typing import Any, Union


class A:
    pass


class B:
    pass


x = Any
y = Any
a = A
u = Union
assert x is Any
assert x is y
assert a is A
assert a is not B
assert u is Union
assert a is not Any
