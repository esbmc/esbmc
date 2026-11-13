import random


def f(flag):
    if flag:
        return 7
    return "seven"


flag = random.random() < 0.5
x = f(flag)
y = 14 // x
