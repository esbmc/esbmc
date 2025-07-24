# 1-Byte
a = 97
b = chr(a)
assert b == 'a'

# 2-Byte
c = 200
d = chr(c)
assert d == 'È'

# 3-Byte
e = 23383
f = chr(e)
assert f == '字'

# 4-Byte
g = 65536
h = chr(g)
assert h == '𐀀'