def pick(fl):
    return "s" if fl else 3


flag = nondet_bool()
r = pick(flag)
if flag:
    assert r == "s"
else:
    assert r == 3
