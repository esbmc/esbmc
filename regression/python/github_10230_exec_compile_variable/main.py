code = compile("def f():\n    return 1\n", "<string>", "exec")
exec(code)
assert f() == 1
