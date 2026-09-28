# enumerate shares zip's lowering: a standalone enumerate used to reach the
# generic call builder and yield a wrong list (#7555). Covers the literal fold
# and the name operand, each with a default and an explicit start.

e = list(enumerate([7, 8]))
assert len(e) == 2
assert e[1][0] == 1
assert e[1][1] == 8

s = list(enumerate([7, 8], 5))
assert s[0][0] == 5
assert s[1][0] == 6

xs = [10, 20, 30]
f = list(enumerate(xs))
assert len(f) == 3
assert f[2][0] == 2
g = list(enumerate(xs, 4))
assert g[0][0] == 4
