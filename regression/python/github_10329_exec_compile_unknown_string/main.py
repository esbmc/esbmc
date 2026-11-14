import random

part = "2" if random.random() < 0.5 else "3"
if random.random() < 0.5:
    code = "a = 1"
else:
    code = "a = " + part
exec(compile(code, "<string>", "exec"))
assert a == 1 or a == 2 or a == 3
