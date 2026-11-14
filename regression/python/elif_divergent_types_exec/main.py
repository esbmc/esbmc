import random

n = random.randint(0, 10)
is_str = random.random() < 0.5
if is_str:
    code = "a = 'Hello'"
else:
    code = "a = n + 1"
exec(code)
b = a
if is_str:
    assert b == "Hello"
else:
    assert isinstance(b, int)
    assert b == n + 1
