# Dynamic retyping inside a conditional body. A variable bound to a numeric
# scalar is rebound to a string (and back) inside an `if` block. Reads INSIDE
# the block must observe the new (string) type, while the variable's float
# binding established after the block is unaffected. Previously the frontend
# coerced the in-block string write to a numeric value (0.0) and reported a
# spurious counterexample for the in-block assertion.
a = 1
assert a == 1
if True:
    a = "Rafael"
    assert a == "Rafael"
a = 1.3
assert a == 1.3
