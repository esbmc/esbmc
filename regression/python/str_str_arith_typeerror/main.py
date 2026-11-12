a = "6"
b = "3"
raised = 0

try:
    a / b
except TypeError:
    raised += 1

try:
    a - b
except TypeError:
    raised += 1

try:
    a // b
except TypeError:
    raised += 1

try:
    a**b
except TypeError:
    raised += 1

assert raised == 4
assert a + b == "63"
assert b < a
