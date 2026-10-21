def second(ref):
    x, y = ref
    return y


d = {"a": (1, 2), "b": (3, 4)}
v = list(d.values())
assert second(v[0]) == 2
assert second(v[1]) == 3
