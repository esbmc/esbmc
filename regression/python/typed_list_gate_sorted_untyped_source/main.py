a = []
a.append(2.5)
a.append(1.5)
s = sorted(a)
s.append(3)
assert s[0] == 1.5
assert s[2] == 3
