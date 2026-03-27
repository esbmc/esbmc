def f():
    ret = []
    for r in [[]]:
        ret.extend([1] + r)
    return ret


f()
