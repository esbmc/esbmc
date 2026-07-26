# Regression for issue #5904: a missing method on a builtin value must raise a
# catchable AttributeError, not a generic "Unsupported function" assertion.
def main():
    x = 5
    x.foo()


main()
