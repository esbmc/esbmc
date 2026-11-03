import random

r = random.randint(0, 1) == 1
k = [3, 1, 2]
k.sort(reverse=r)
if r:
    assert k == [3, 2, 1]
else:
    assert k == [1, 2, 3]
m = [3, 1, 2]
m.sort(key=lambda x: -x, reverse=r)
if r:
    assert m[0] == 1
else:
    assert m[0] == 3
