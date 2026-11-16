import random


class A:
    pass


a = A()
flag = random.random() < 0.5
if flag:
    a.code = 1
assert hasattr(a, "code") == flag
