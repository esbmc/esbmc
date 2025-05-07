# Unicode characters
unicode1 = str("café")
unicode2 = str("café")
assert unicode1 == unicode2
assert unicode1 != "cafe"

# Strings with null characters
null_str1 = str("abc\0def")
null_str2 = str("abc\0def")
assert null_str1 == null_str2
assert null_str1 != "abcdef"

# Whitespace strings
ws1 = str(" ")
ws2 = str(" ")
assert ws1 == ws2
assert ws1 != ""

# Identity vs equality
a = str("duplicate")
b = str("duplicate")
assert a == b          # same content, different objects

# Empty string comparisons
empty = str("")
assert empty == ""
assert empty != "nonempty"
