# A tag can hold a float, so isinstance() against float is a real runtime
# check on its type_id rather than a refusal. Issue #7075.

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
x = 1.5
assert isinstance(x, float)
assert not isinstance(x, int)
assert not isinstance(x, str)
