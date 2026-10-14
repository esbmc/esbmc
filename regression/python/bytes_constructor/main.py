# bytes() constructor. bytes(iterable-of-ints) previously built the argument
# list and then relabelled its (pointer) type as the bytes array type without
# converting the value, tripping a base_type_eq assertion in value_set (a
# crash). bytes(n) produced a wrong verdict the same way. Both now build a real
# byte array, like a b"..." literal.
b = bytes([1, 2, 3])
assert len(b) == 3
assert b[0] == 1 and b[1] == 2 and b[2] == 3

c = bytes([10, 20, 30])
assert c[2] == 30
assert len(c) == 3

# bytes(n) -> n zero bytes
z = bytes(3)
assert len(z) == 3
assert z[0] == 0 and z[2] == 0

# byte literals still work
assert b"abc"[1] == 98
assert bytes([65])[0] == 65
