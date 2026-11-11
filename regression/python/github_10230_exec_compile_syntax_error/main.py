ok = True
try:
    code = compile("a =", "<string>", "exec")
except SyntaxError:
    ok = False
if ok:
    exec(code)
assert not ok
