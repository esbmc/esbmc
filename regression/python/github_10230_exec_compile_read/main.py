code = compile("a = 5", "<string>", "exec")
exec(code)
assert a == 5
assert code is not None
