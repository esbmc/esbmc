# A tag holding a float keeps its type through a list round-trip. Comparing a
# tag against a float literal is not supported yet, so this pins the type_id.
cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
if cond:
    x = 1.5
lst = []
lst.append(x)
assert isinstance(lst[0], float) or isinstance(lst[0], str)
