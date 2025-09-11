a = 2
b = 4
c = 8

# Exponentiation
assert a**b == 16
x = -a**b
assert x == -16  # Unary minus binds less tightly than **
assert (-a)**b == 16
assert x is not b

# Unary operations
assert -a == -2
assert +a == 2
assert ~a == -3
assert not False is True
assert not True is False

# Multiplication, Division, Floor Division, Modulo
assert a * b == 8
assert c / b == 2.0
assert c // b == 2
assert c % b == 0
assert 3 * 2 % 4 == 2  # * binds tighter than %
assert a * b is 8

# Addition and Subtraction
assert a + b == 6
assert c - a == 6
assert a + b * c == 34  # * has higher precedence than +
assert (a + b) * c == 48

# Mixed arithmetic and bitwise
assert 1 << 2 + 1 == 8  # 2 + 1 = 3; 1 << 3 = 8
assert (1 << 2) + 1 == 5

# Bitwise Operations
assert a | b == 6
assert a ^ b == 6
assert a & b == 0
assert b >> 1 == 2
assert a << 1 == 4
assert 1 | 2 & 3 == 3  # & binds tighter than |
assert (1 | 2) & 3 == 3

# Comparison Operations
assert b > a
assert c >= b
assert a < b
assert a <= c
assert a != b
assert b == 4
assert 2 + 2 == 4
assert 2 + 2 > 3 and 1 < 2

# Assignment Operators
x = a
x += 2
assert x == 4
x -= 1
assert x == 3
x *= 2
assert x == 6
x //= 2
assert x == 3
x /= 3
assert x == 1.0
x %= 1
assert x == 0
x = 2
x **= 3
assert x == 8
x &= 6
assert x == 0
x |= 7
assert x == 7
x ^= 2
assert x == 5
x <<= 1
assert x == 10
x >>= 2
assert x == 2

# Logical Operators
assert not False
assert a == 2 and b == 4
assert a == 2 or c == 100
assert not (a != 2 or b != 4)
assert a == 2 and (b > 3 or c < 2)
assert a < b and b < c and c == 8
