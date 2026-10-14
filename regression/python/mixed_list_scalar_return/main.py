# An unannotated function with a heterogeneous (union) return: a list on one
# branch (str.split()) and a scalar on the other (len()). The annotation pass
# previously inferred only the scalar branch's type (split() was not recognised
# as list-returning), annotating the whole function `-> int`. The call result
# was then typed as a scalar and `r = f(...)` const-folded to a bogus size-1
# list, so `len(r)` collapsed to 1. Recognising split() as list-returning makes
# the inferred return set {list, int} -> left unannotated -> each return path is
# typed correctly, so the list branch yields a real runtime list.


def f(txt):
    if " " in txt:
        return txt.split()
    else:
        return len(txt)


r = f("Hello world!")
assert len(r) == 2
assert r == ["Hello", "world!"]
