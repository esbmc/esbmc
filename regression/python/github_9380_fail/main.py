# Issue #9380: divmod() by a float zero raises ZeroDivisionError.
a: float = 1.0
b: float = 0.0
q: float = divmod(a, b)[0]
