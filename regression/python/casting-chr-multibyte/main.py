# 1-Byte
a = chr(97)
assert a == 'a'

# 2-Byte
b = chr(200)
assert b == 'È'

# 3-Byte
c = chr(23383)
assert c == '字'

# 4-Byte
d = chr(65536)
assert d == '𐀀'