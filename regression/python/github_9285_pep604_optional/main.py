# An `int | None` parameter returned beside a str literal (#9285).
from mod import func, gunc, hunc

assert func() == "a"
assert func(4) == 4
assert gunc() == "a"
assert gunc(4) == 4
assert hunc(3) == 3
