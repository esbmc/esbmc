# Rebinding a tagged variable to a container gives the name a fresh slot of the
# container's type, so the list is fully usable afterwards. Issue #7075: this
# used to abort the frontend.

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
x = [1, 2, 3]
assert x[1] == 2
assert len(x) == 3
