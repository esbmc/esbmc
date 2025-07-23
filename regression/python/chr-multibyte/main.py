# 1-Byte
a = chr(97)
assert a == 'a'

# 2-Byte
a = chr(200)
assert a == 'È'

# 3-Byte
a = chr(23383)
assert a == '字'

# 4-Byte
a = chr(65536)
assert a == '𐀀'