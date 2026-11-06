# GitHub #9929: a str-returning function whose returned name is first bound
# from a list element was typed as returning int.
def g(a, b):
    xs = [a, b]
    best = xs[0]
    if len(xs) > 5:
        best = b
    return best


def f(a, b):
    return max(a, b, key=len)


assert g("bb", "a") == "bb"
r = f("a", "bb")
assert r != "bb"
