t = (1, 2, 3)
try:
    t[0] = 100
    assert False, "Tuple should be immutable!"
except TypeError:
    pass  # Expected behavior
