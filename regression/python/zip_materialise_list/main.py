# zip is modelled as a for-loop rewrite, so as a standalone value it used to
# yield a list of the wrong length and elements (#7555). Covers both operand
# shapes the rewrite handles -- literals fold to a list of tuples, names lower
# to an indexed comprehension -- each with more than two operands and with
# truncation to the shortest input, which is the easiest half to get wrong.

v = list(zip([1, 2], [3, 4]))
assert len(v) == 2
assert v[0][0] == 1
assert v[1][1] == 4

t = list(zip([1, 2], [3, 4], [5, 6]))
assert len(t) == 2
assert t[1][2] == 6

assert len(list(zip([1, 2, 3], [4, 5]))) == 2

xs = [1, 2, 3]
ys = [4, 5]
w = list(zip(xs, ys))
assert len(w) == 2
assert w[1][1] == 5

zs = [7]
u = list(zip(xs, ys, zs))
assert len(u) == 1
assert u[0][2] == 7
