# Regression for esbmc/esbmc#5102: copying a list must preserve the values of
# nested-list elements. The per-element copy used to byte-copy the inner list
# pointer's pointee, corrupting nested elements on concat / copy() / list() /
# repeat / variable assignment.
a = nondet_int()
__ESBMC_assume(a >= 0)
__ESBMC_assume(a <= 9)

src = [[a], [7]]

# concatenation
cat = [[a]] + [[7]]
assert cat[0][0] == a
assert cat[1][0] == 7

# list.copy() and list()
c1 = src.copy()
c2 = list(src)
assert c1[0][0] == a
assert c2[1][0] == 7

# repeat-then-concat
rep = [[1], [2]] + [[a]]
assert rep[2][0] == a

# variable assignment aliases the same nested elements
alias = src
assert alias[0][0] == a
