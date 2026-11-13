import random


class A:
    pass


a = A()
if random.random() < 0.5:
    a.code = 1
del a.code
