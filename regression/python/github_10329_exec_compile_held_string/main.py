import random

if random.random() < 0.5:
    code = "a = 'x'"
else:
    code = "a = 1"
exec(compile(code, "<string>", "exec"))
b = a
assert b == "x" or b == 1
