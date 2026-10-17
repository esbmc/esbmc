# x joins a nondet int with a str literal; adding 1 to it should raise
# a TypeError on the str branch.
n = nondet_int()
if n > 0:
    x = n
else:
    x = "a"
y = x + 1
assert y == n + 1
