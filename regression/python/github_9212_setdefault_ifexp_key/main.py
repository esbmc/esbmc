# setdefault with a str-literal-or-nondet_str key (#9212).
d = {"a": 0}
k = "a" if nondet_bool() else nondet_str()
d.setdefault(k, 1)
assert d["a"] == 0
