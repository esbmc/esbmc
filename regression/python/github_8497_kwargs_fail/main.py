def cat(**kw):
    r = ""
    for v in kw.values():
        r += v
    return r


assert cat(b="y", a="x") == "xy"
