def f(a):
    if not a:
        return ''
    return a[0]

assert f("ab") == "a"
