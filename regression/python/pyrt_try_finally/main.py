# `finally` runs however the region is left. Each path adds a distinct amount,
# so a cleanup that ran twice or not at all shows up as a different total.
class E(Exception):
    pass


class F(Exception):
    pass


# No handler at all.
log = 0
try:
    log = 1
finally:
    log = log + 10
assert log == 11

# A handler took it.
log = 0
try:
    raise E()
except E:
    log = 1
finally:
    log = log + 10
assert log == 11

# else and finally together, in order.
log = 0
try:
    log = 1
except E:
    log = 99
else:
    log = log + 10
finally:
    log = log + 100
assert log == 111
