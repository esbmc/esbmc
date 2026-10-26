# setdefault with a str-literal-or-nondet_str key can add a key (#9212).
d = {"a": 0}
k = "a" if nondet_bool() else nondet_str()
d.setdefault(k, 1)
assert len(d) == 1
