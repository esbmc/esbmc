x:int = 0
try:
    # some code
    x = x + 1
    if x > 0:
        raise TypeError("Test exception")
except ValueError:      # name is null - no variable binding
    x = -1
except TypeError as e:  # name is "e" - has variable binding
    x = 100

assert x <= 50, "x should be <= 50"  # Will fail since x becomes 100
