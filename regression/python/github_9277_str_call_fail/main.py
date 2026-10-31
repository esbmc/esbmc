# As github_9277_str_call, but the str path asserts what is false.
n = abs(nondet_int()) % 3
c = nondet_bool()
k = n if c else str(n)
if c:
    assert isinstance(k, int)
else:
    assert isinstance(k, int)
