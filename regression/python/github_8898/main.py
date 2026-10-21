# A list mixing int and str, passed through an unannotated parameter, must
# compare equal to an equal list (#8898).
def check(x):
    assert x == [10, "-"]


def check_pair(x, y):
    assert x == y


def check_not_equal(x, y):
    assert x != y


check([10, "-"])
check_pair([10, "-"], [10, "-"])
check_not_equal([10, "-"], ["-", 10])
