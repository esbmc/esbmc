# github #10499: a rebound builtin name is no longer callable
int = 5
len = 3
assert not callable(int)
assert not callable(len)
