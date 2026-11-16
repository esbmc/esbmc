raised = False
try:
    sorted([1, "a"])
except TypeError:
    raised = True
assert raised
