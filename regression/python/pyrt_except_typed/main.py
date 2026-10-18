# A typed handler matches on the exception's class at run time, so a subclass
# is caught by a handler naming its base, and the handlers are tried in order.
class Outer(Exception):
    pass


class Inner(Outer):
    pass


class Other(Exception):
    pass


c = 0
try:
    raise Inner()
except Outer:
    c = 1
assert c == 1

c = 0
try:
    raise Other()
except Inner:
    c = 1
except Other:
    c = 2
assert c == 2

# A bare handler after a typed one takes whatever is left.
c = 0
try:
    raise Other()
except Inner:
    c = 1
except:
    c = 2
assert c == 2
