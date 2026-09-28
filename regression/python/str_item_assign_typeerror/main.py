# Assigning through a string subscript is a TypeError: strings are immutable.
# Caught here rather than updating the char array with a whole string value,
# which tripped with2t::assert_consistency and aborted.
s = "hello"

try:
    s[0] = "H"
    caught = False
except TypeError:
    caught = True

assert caught
assert s == "hello"
