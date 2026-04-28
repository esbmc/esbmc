def g(*, items=[]):
    return 7


assert g() == 7
assert g(items=[1]) == 7
