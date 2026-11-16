# Issue #9382: float `%` by zero must raise ZeroDivisionError.
a: float = 1.0
r: float = a % 0.0
