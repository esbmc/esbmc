x = 0
try:
    # some code
    x = x + 1
    raise ValueError("Test exception")
except ValueError:  # name is null - no variable binding
    assert False, "This assertion will fail"  # Will cause failure
except TypeError as e:  # name is "e" - has variable binding
    pass
