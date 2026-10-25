# github #9048: object() must give a fresh object, never None.
x = object()
assert x is not None

y = object()
assert x is not y

objs = []
for _ in range(2):
    objs.append(object())
assert objs[0] is not objs[1]
