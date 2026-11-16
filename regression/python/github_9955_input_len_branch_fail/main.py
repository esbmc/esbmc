if nondet_bool():
    s = "abc"
else:
    s = input()
assert len(s) == 2 or s != "abc"
