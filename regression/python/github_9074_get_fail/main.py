# 9074: a returned dict.get without a default may be None
def f(**kw):
    return kw.get('error')


r = f()
assert r is not None
