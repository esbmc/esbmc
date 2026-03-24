def f(a):
    if not a:
        return ''
    return a[0] + f(a[1:])

assert f("ab") == "ab"
