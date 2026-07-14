# Negative variant of nested-list-return: the float element returned from a
# function is read at runtime from the caller, so an assertion on the wrong
# value must FAIL (proves the read is genuine, not vacuous).


def build_float():
    return [[1.5]]


Qf = build_float()
assert Qf[0][0] == 2.5
