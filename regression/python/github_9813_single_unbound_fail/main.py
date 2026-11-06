import random


class A:
    def f1(self):
        return 1


a = A()
if random.random() < 0.0:
    c = a.f1
x = c()
assert x == 1
