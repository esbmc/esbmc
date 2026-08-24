# Runtime type diverges across branches (int vs str). '-'/'/' between two
# such variables works now, but '+' still refuses with a clean error, since
# a genuine str+str path would need string concatenation between two
# tagged operands (unlike '-'/'/', which never apply to str in Python).
# Correct verdict, once '+' is supported: VERIFICATION FAILED ("ab" == 3 is
# False).
cond = nondet_bool()
if cond:
    x = 1
    y = 2
else:
    x = "a"
    y = "b"
z = x + y
assert z == 3
