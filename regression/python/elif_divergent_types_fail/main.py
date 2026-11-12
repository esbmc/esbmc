import random

c = random.random() < 0.5
if c:
    a = "Hello"
elif not c:
    a = 5
if c:
    assert a == "Bye"
else:
    assert isinstance(a, str)
