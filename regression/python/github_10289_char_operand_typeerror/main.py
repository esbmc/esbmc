s = "ab"
t = "cd"
raised = 0

try:
    s - t[1]
except TypeError:
    raised += 1

try:
    t[0] - t[1]
except TypeError:
    raised += 1

try:
    t[1] // s
except TypeError:
    raised += 1

try:
    t[1] + 1
except TypeError:
    raised += 1

try:
    2**t[0]
except TypeError:
    raised += 1

try:
    t[0] & t[1]
except TypeError:
    raised += 1

try:
    s * t[1]
except TypeError:
    raised += 1

assert raised == 7
assert s + t[1] == "abd"
assert t[0] < t[1]
assert ord(t[1]) - ord(t[0]) == 1
assert chr(ord(t[0]) + 1) == "d"
