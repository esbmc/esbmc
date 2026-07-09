# Regression for issue #5904: a missing attribute read must raise a catchable
# AttributeError (previously aborted frontend conversion).
def main():
    x = 5
    y = x.foo


main()
