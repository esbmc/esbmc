import random

if random.random() < 0.5:
    code = "a = 2"
else:
    code = "a = 1"
code_obj = compile(code, "<string>", "exec")
exec(code_obj)
b = a
if code == "a = 2":
    assert b == 2
else:
    assert b == 1
