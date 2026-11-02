# #9365: str of a tuple is not modelled, so f"{t}" is any string.
t = (1,)
assert f"{t}" != "(1,)"
