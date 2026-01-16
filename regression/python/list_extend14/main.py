def f() -> list[list]:
    return [[]]

def g():
    ret = []
    ret.extend([1] + r for r in f())
    return ret

g()
