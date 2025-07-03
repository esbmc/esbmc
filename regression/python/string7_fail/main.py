# Unicode mismatch
assert str("café") != "café"

# Null char mismatch
assert str("abc\0def") != "abc\0def"

# Whitespace string failure
assert str(" ") != " "

# Empty vs empty
assert "" != str("")
