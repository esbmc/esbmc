def pick(x, c):
    if c:
        return x
    return 0


flag = nondet_bool()
v = nondet_int() if flag else nondet_str()
r = pick(v, nondet_bool())
assert r == v or r == 0
