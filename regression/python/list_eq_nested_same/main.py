# Companion (currently-correct) coverage for nested-list equality: element,
# inner-list, and whole-list comparisons all hold when both operands are built
# the same way. Pins the behaviour that list_eq_nested_comprehension isolates
# as broken only across mixed construction (comprehension vs literal).
m = [[i * j for j in range(2)] for i in range(2)]
n = [[i * j for j in range(2)] for i in range(2)]
assert len(m) == 2
assert m[1] == [0, 1]      # inner-list equality holds
assert m == m              # same object
assert m == n              # two comprehensions, same type representation
