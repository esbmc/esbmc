# str % with a subscript operand lowers like the matching f-string (#8672).
def f(xs, i):
    return "v %d" % xs[i]


def g(xs: list[int], i: int) -> str:
    try:
        return "v %d" % xs[i]
    except IndexError:
        return "none"


names = ["a", "bc"]
d = {"k": 7}
t = (1, "x")
assert f([1, 2], 1) == "v 2"
assert g([1, 2], 2) == "none"
assert g([1, 2], 0) == "v 1"
assert "%s=%d" % (names[1], d["k"]) == "a=7"
assert "%d-%s" % (t[0], t[1]) == "1-x"
