# Arithmetic combining a typed scalar with a value read from an
# unannotated nested-list parameter used to crash the frontend with a
# bv-width assertion (#4830). Call-site inference must give `items` the
# fully nested annotation list[list[int]] so the inner element resolves
# to int instead of Any.
def f(items, x):
    return x - items[0][0]


assert f([[3]], 1) == -2
