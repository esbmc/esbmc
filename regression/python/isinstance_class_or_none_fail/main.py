# 10434: an instance-or-None name that holds None is not an instance.
import random


class A:
    pass


y = A() if random.random() < 0.0 else None
assert isinstance(y, A)
