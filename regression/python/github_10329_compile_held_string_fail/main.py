import random

if random.random() < 0.5:
    code = "a = 'x'"
else:
    code = "a = 1"
code_obj = compile(code, "<string>", "exec")
exec(code_obj)
b = a
assert b == "y" or b == 2
