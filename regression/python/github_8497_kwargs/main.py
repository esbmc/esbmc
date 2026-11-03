def cat(**kw):
    r = ""
    for v in kw.values():
        r += v
    return r


assert cat(a="x", b="y") == "xy"
assert cat(b="y", a="x") == "yx"
