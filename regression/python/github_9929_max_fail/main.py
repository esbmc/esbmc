# GitHub #9929: max(..., key=len) in a function is lowered to a scan that
# returns a name bound from a list element.
def f(a, b):
    return max(a, b, key=len)


r = f("a", "bb")
assert r != "bb"
