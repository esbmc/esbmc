import re

raised = False
try:
    re.sub("a", "b", 5)
except TypeError:
    raised = True
assert raised
