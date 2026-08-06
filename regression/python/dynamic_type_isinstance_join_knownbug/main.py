# isinstance() cannot yet tell a variable's real type when it differs per
# branch, so ESBMC refuses with a clean error. Correct verdict, once
# supported: VERIFICATION SUCCESSFUL (the assertion always holds on the
# string path).

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
if not cond:
    assert not isinstance(x, int)
