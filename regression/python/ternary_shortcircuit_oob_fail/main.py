# The selected branch of a conditional expression is still bounds-checked: the
# taken branch reads a[5] out of range for the 1-element list, raising a
# catchable IndexError that goes uncaught here.
def f() -> int:
    a = [10]
    c = 5
    return a[c] if c > 0 else 0


f()
