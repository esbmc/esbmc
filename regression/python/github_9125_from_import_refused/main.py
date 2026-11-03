# mod.py returns f (str) or g (int) that it imports from mod2 (#9125).
from mod import pick

flag = True
a = pick(flag)
b = a()
if flag:
    assert b == "a"
else:
    assert b == 1
