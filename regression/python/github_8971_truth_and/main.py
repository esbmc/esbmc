flag = nondet_bool()
other = nondet_bool()
x = nondet_int() if flag else nondet_str()
truthy = (isinstance(x, int) and x != 0) or (isinstance(x, str) and x != "")
if x and other:
    assert other and truthy
else:
    assert not other or not truthy
r = 1 if x else 0
assert (r == 1) == truthy
