import random

if random.random() < 0.5:
    code = "a = 5"
else:
    code = "a = 6"
exec(code)
b = a
assert b == 7
