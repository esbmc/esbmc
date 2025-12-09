x : int = 0
y : int = 0
if (x == 1 and (5/y) > 0):  # Should short-circuit, not divide by zero
    result: float = 1/0
