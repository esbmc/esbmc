import random

flag = random.random() < 0.5
a = [[1]]
a.append(a)
b = [[3]]
b.append(b)
ls = a if flag else b
assert len(ls) == 2
