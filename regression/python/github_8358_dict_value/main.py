# An unannotated parameter fed a dict lookup takes the dict's value type,
# so unpacking it reads a tuple rather than an untyped pointer.


def f(ref):
    x, y = ref
    return y


d = {'a': (1, 2)}
assert f(d['a']) == 2
