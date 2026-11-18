def f(**kw):
    return kw.get('error')


assert f() is None
