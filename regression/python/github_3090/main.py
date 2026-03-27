# Valid cases
assert chr(65) == 'A'  # Basic ASCII
assert chr(97) == 'a'  # Lowercase ASCII
assert chr(8364) == '€'  # Unicode Euro sign
assert chr(0x1F600) == '😀'  # Emoji

# Edge cases
assert chr(0) == '\x00'  # Null character
assert chr(0x10FFFF) == '\U0010FFFF'  # Highest valid Unicode code point

# Invalid cases: should raise ValueError
try:
    chr(-1)
except ValueError:
    pass
else:
    raise AssertionError("chr(-1) should raise ValueError")

try:
    chr(0x110000)
except ValueError:
    pass
else:
    raise AssertionError("chr(0x110000) should raise ValueError")
