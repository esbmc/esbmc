# Issue #10222: the class body statement raising ZeroDivisionError was dropped,
# so the program verified.
def f(pow=None):

    class A:
        x = pow(0, -1)
        pow = None


f()
