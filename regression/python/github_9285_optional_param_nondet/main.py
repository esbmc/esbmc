# An Optional[int] parameter returned beside a str literal (#9285).
from mod import func

v = nondet_int()
assert func(v) == v
assert func() == "a"
