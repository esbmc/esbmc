# GitHub #9088: list + tuple raises TypeError; list + list concatenates.

caught = False
try:
    x = [1] + (2, 3)
except TypeError:
    caught = True
assert caught

y = [1] + [2, 3]
assert len(y) == 3
