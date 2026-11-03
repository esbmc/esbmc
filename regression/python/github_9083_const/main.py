c = chr(0xD800)
assert ord(c) == 0xD800
assert c != chr(0xDC00)
assert c == chr(55296)
