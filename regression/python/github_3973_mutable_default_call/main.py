def use_default(xs=[]):
    return 1


assert use_default() == 1
assert use_default([1, 2]) == 1
