a = "6"
raised = 0

try:
    a -= 1
except TypeError:
    raised += 1

try:
    a /= 2
except TypeError:
    raised += 1

assert raised == 2
assert a == "6"
a += "3"
assert a == "63"
