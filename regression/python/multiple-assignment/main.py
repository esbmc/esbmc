a = b = 0
a += 1
b += 2
assert a == 1
assert b == 2

x = y = 5
x += 3
y += 7
assert x == 8
assert y == 12

m = n = 10
m -= 4
n *= 2
assert m == 6
assert n == 20

p = q = 8
p //= 2
q += 10
assert p == 4
assert q == 18

u = v = 3
u **= 2
v -= 1
assert u == 9
assert v == 2

r = s = 15
r %= 4
s -= 5
assert r == 3
assert s == 10

flag1 = flag2 = True
flag1 = not flag1
flag2 = False
assert flag1 == False
assert flag2 == False
