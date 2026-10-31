from mod import make

m = make()
assert m.get() == 1
f = m.me()
assert f.get() == 1
