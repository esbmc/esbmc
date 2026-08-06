# Runtime type diverges across branches (int vs str). ESBMC does not yet
# support arithmetic on a variable whose type differs per branch, so it
# refuses with a clean error. Correct verdict, once supported: VERIFICATION
# FAILED ("ab" == 3 is False).
cond = nondet_bool()
if cond:
    x = 1
    y = 2
else:
    x = "a"
    y = "b"
z = x + y
assert z == 3
