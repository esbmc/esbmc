def f(ref):
    x, y = ref
    return y


d = {'a': (1, 2)}
assert f(d['a']) == 1
