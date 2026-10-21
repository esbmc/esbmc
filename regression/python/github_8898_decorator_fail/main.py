# The decorator calls check with a list its call sites do not show, so the
# parameter must not be typed from them (#8898).
def deco(fn):
    def wrapper(a):
        return fn([10, "+"])

    return wrapper


@deco
def check(x):
    assert x == [10, "-"]


check([10, "-"])
