def one():
    return 42


call = lambda h: h()
flag = True
r = call(one if flag else "abc")
assert r == 43
