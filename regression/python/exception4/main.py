x = 0
try:
    # some code  
    x = x + 1
    raise TypeError("Forced exception")
except ValueError:      # name is null - no variable binding
    x = x * 10  # x becomes 10
except TypeError as e:  # name is "e" - has variable binding
    pass

assert x < 5  # Will succeed since x is 1
