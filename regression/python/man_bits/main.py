a = 10  # 1010 in binary
b = 5  # 0101 in binary

# Bitwise OR (|)
assert (a | b) == 15, "a | b should be 15"  # 1010 | 0101 = 1111 (15)

# Bitwise AND (&)
assert (a & b) == 0, "a & b should be 0"  # 1010 & 0101 = 0000 (0)

# Bitwise NOT (~)
assert (~a) == -11, "~a should be -11"  # ~1010 = -(1010 + 1) = -11
assert (~b) == -6, "~b should be -6"  # ~0101 = -(0101 + 1) = -6

# Bitwise XOR (^)
assert (a ^ b) == 15, "a ^ b should be 15"  # 1010 ^ 0101 = 1111 (15)

# Left Shift (<<)
assert (a << 1) == 20, "a << 1 should be 20"  # 1010 << 1 = 10100 (20)
assert (b << 2) == 20, "b << 2 should be 20"  # 0101 << 2 = 10100 (20)

# Right Shift (>>)
assert (a >> 1) == 5, "a >> 1 should be 5"  # 1010 >> 1 = 0101 (5)
assert (b >> 1) == 2, "b >> 1 should be 2"  # 0101 >> 1 = 0010 (2)
