# A function also called through an alias is not specialized (#9545): the
# alias call would reach a copy typed for another call's elements.
def f(xs):
    return xs[0]


g = f
f(["a", "b"])
f([4, 5, 6])
assert g([1.5]) != 1.5
