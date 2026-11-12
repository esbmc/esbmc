import random

k = random.randint(0, 2)
a = 7
if k == 0:
    a = "Hello"
elif k == 1:
    a = 5
if k == 0:
    assert a == "Bye"
elif k == 1:
    assert a == 6
else:
    assert a == 8
