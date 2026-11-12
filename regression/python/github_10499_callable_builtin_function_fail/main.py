# github #10499: builtin functions are callable
assert callable(len)
assert not callable(print)
