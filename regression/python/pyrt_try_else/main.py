# `else` runs only when the body completed: an exception leaves the body and
# skips whatever follows it.
class E(Exception):
    pass


log = 0
try:
    log = 1
except E:
    log = 99
else:
    log = log + 10
assert log == 11

log = 0
try:
    raise E()
except E:
    log = 1
else:
    log = 99
assert log == 1
