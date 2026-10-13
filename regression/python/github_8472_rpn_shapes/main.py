# The calls pass floats and strings in different positions; a tagged read
# answers for any position, so the shapes need not agree (#8472).
def count_floats(xs):
    n = 0
    for x in xs:
        if isinstance(x, float):
            n += 1
    return n


assert count_floats([1.0, "+"]) == 1
assert count_floats([1.0, "+", 2.0]) == 2
