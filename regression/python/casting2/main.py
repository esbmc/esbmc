a = '60'
b = '5'

# Integer conversion
assert int(a) == 60
assert int(b) == 5

# Float conversion
assert float(a) == 60.0
assert float(b) == 5.0

# String conversion
#assert str(60) == '60'
#assert str(5.5) == '5.5'

# Character conversion (ASCII values)
assert chr(60) == '<'
assert chr(5) == '\x05'

# Ordinal conversion (character to ASCII/unicode integer)
assert ord('<') == 60
assert ord('5') == 53

# Hexadecimal conversion
assert hex(60) == '0x3c'
assert hex(5) == '0x5'

# Octal conversion
assert oct(60) == '0o74'
assert oct(5) == '0o5'
