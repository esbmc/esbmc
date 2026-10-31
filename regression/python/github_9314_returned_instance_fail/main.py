from mod import make

m = make()
f = m.me()
assert f.get() == 2
