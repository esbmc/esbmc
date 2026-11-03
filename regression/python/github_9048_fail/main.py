# github #9048: two object() calls are distinct objects.
objs = []
for _ in range(2):
    objs.append(object())
assert objs[0] is objs[1]
