# Issue #10222: zero to a negative power raises ZeroDivisionError.
class A:
    x = pow(0, -1)
    pow = None
