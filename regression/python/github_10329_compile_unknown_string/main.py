import random

part = "2" if random.random() < 0.5 else "3"
if random.random() < 0.5:
    code = "a = 1"
else:
    code = "a = " + part
code_obj = compile(code, "<string>", "exec")
exec(code_obj)
assert a == 1 or a == 2 or a == 3
