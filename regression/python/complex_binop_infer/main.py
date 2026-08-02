# An unannotated complex value formed by `<real> + <imaginary>j`. The type
# inference for a binary expression was driven by the left operand only, so
# `3 + 4j` was typed int (from the `3`) and `.real`/`.imag` access then raised
# AttributeError. `4j + 3` happened to work because the complex operand was on
# the left. Python's numeric tower promotes to complex when either operand is
# complex; the inference now reflects that.

z = 3 + 4j
assert z.real == 3.0
assert z.imag == 4.0

# Complex on the left still works, and arithmetic stays complex.
w = 4j + 3
assert w.real == 3.0

p = 5 + 2j
q = p * 2
assert q.real == 10.0
assert q.imag == 4.0
