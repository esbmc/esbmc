# The cleanup runs on the paths that leave the region with an exception still
# in flight: one no handler matched, and one a handler raised itself.
class E(Exception):
    pass


class F(Exception):
    pass


ran = 0
try:
    try:
        raise E()
    except F:
        ran = 99
    finally:
        ran = 1
except E:
    ran = ran + 10
assert ran == 11

ran = 0
try:
    try:
        raise E()
    except E:
        raise F()
    finally:
        ran = 1
except F:
    ran = ran + 10
assert ran == 11
