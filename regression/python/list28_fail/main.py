xs = [1.0, 2.0, 3.0]
s = 0.1
while xs:
    s = s + xs.pop()
assert s == 6.0
