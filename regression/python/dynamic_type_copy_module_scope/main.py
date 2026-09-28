# Copying a tagged name at module scope aborted in member2t: the RHS type probe
# that adopts the tagged type runs only in function scope.
cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
y = x
z = y
assert z == 1 or z == "a"
