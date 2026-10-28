flag = nondet_bool()
p = nondet_int() if flag else nondet_str()
if isinstance(p, str):
    if p != "":
        h = p[0]
        assert h == p[0]
        last = p[-1]
        assert last == p[-1]
