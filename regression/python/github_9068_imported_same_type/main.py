# pick() returns functions that both return str, so the call is converted.
from mod import pick

b = pick(True)()
assert b == "a"
