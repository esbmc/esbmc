if nondet_bool():
    s = "abc"
else:
    s = input()
assert len(s) == 3 or s != "abc"
