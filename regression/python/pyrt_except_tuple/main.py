# `except (A, B):` matches when any of the named classes does, and lets
# anything else keep propagating.
class E(Exception):
    pass


class F(Exception):
    pass


class G(Exception):
    pass


c = 0
try:
    raise E()
except (E, F):
    c = 1
assert c == 1

c = 0
try:
    raise F()
except (E, F):
    c = 1
assert c == 1

c = 0
try:
    try:
        raise G()
    except (E, F):
        c = 99
except G:
    c = 1
assert c == 1
