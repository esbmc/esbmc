# 9074: a returned dict.get without a default may be None
def f(**kw):
    return kw.get('error')


def g(k: str):
    d = {'a': 1.5}
    return d.get(k)


def h(x: int, **kw):
    if x > 0:
        return kw.get('n', 7)
    return kw.get('e')


def second_or_none(lst):
    return None if len(lst) < 2 else lst[1]


assert f() is None
assert f(x=1) is None
assert f(error=3) == 3
assert g('b') is None
assert g('a') == 1.5
assert second_or_none([1]) is None
assert second_or_none([1, 2]) == 2
assert h(x=1) == 7
