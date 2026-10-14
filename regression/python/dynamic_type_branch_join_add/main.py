# Runtime type diverges across branches (int vs str); x + y's own result
# type is only known at runtime too, so z becomes tagged. z == 3 holds on
# the int path but fails on the str path ("ab" == 3 is False).
cond = nondet_bool()
if cond:
    x = 1
    y = 2
else:
    x = "a"
    y = "b"
z = x + y
assert z == 3
