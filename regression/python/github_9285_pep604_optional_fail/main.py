# An `int | None` parameter returned beside a str literal (#9285).
from mod import func, gunc

assert func(4) == 4
assert gunc(4) == 4
assert func() == "a"
assert gunc() != "a"
