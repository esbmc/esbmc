import random


def f(flag):
    if flag:
        return 0
    return "zero"


flag = random.random() < 0.5
x = f(flag)
if flag:
    y = 7 % x
