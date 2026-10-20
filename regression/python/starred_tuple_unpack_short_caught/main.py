t = (1, )
raised = False
try:
    a, *b, c = t
except ValueError:
    raised = True
assert raised
