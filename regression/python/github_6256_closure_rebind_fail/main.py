# A Python function value is one static symbol here (#6640), so two closures
# over the same def share a capture cell. The second instantiation disagrees
# with the first, which must poison the cell rather than overwrite it: a5(3)
# is 8, so proving it equal to a7's 10 would be unsound.


def make_adder(n):
    def add(x):
        return x + n

    return add


a5 = make_adder(5)
a7 = make_adder(7)
assert a5(3) == 10
