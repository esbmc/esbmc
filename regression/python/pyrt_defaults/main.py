# `defaults` covers the last parameters, so a short call fills from the right:
# h(1, 9) binds b=9 and takes c from its default.
def h(a, b=2, c=3):
    return a * 100 + b * 10 + c


def g(a=1, b=2):
    return a * 10 + b


assert h(1) == 123
assert h(1, 9) == 193
assert h(1, 9, 8) == 198
assert g() == 12
assert g(5) == 52
