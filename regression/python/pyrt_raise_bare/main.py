# A bare `raise` re-raises the exception in flight, and the cleanup of an
# enclosing finally still runs on the way out.
class E(Exception):
    pass


c = 0
try:
    try:
        raise E()
    except E:
        c = 1
        raise
except E:
    c = c + 10
assert c == 11

ran = 0
try:
    try:
        raise E()
    except E:
        raise
    finally:
        ran = 1
except E:
    ran = ran + 10
assert ran == 11
