# Issue #9155: int() of a name holding a str or an int on a nondet path.
c = nondet_bool()
if c:
    r = "a"
else:
    r = 1
y = int(r)
