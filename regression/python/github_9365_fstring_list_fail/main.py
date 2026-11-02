# #9365: str of a list is not modelled, so f"{x}" is any string.
x = [1, 2]
assert f"{x}" != "[1, 2]"
