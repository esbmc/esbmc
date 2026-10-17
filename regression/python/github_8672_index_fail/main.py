# The subscript of a str % operand raises IndexError where Python does.
def f(xs: list[int], i: int) -> str:
    return "v %d" % xs[i]


f([1, 2], 2)
