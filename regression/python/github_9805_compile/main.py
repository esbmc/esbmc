exec(compile("a = 2", "<string>", "exec"))
b = eval(compile("a * 3", "<string>", "eval"))
assert b == 6
