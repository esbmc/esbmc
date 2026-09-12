# The preprocessor unrolls this loop into a chain of tagged-name copies, which
# is why it aborted at module scope and not inside a function.
cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
for e in lst:
    assert e == 1 or e == "a"
