l = [10, 20, 30, 40, 50]
a = l.pop(-1)  # 50
assert a == 50
b = l.pop(-1)  # 40
assert b == 40
assert len(l) == 3
assert l[2] == 30
