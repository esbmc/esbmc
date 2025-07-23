# 1-Byte
a = 97
b = chr(a)
assert b == 'a'

# 2-Byte
a = 200
b = chr(a)
assert b == 'È'

# 3-Byte
a = 23383
b = chr(a)
assert b == '字'

# 4-Byte
a = 65536
b = chr(a)
assert b == '𐀀'