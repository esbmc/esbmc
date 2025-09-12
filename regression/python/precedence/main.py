a = 2
b = 4
c = 8

# Exponentiation
assert a ** b == 16

# Unary operations
assert -a == -2
assert ~a == -3

# Multiplication, Division, Floor Division, Modulo
assert a * b == 8
assert c / b == 2.0
assert c // b == 2
assert c % b == 0

# Addition and Subtraction
assert a + b == 6
assert c - a == 6

# Bitwise Operations
assert a | b == 6
assert a ^ b == 6
assert a & b == 0
assert b >> 1 == 2
assert a << 1 == 4

# Comparison Operations
assert b > a
assert c >= b
assert a < b
assert a <= c
assert a != b
assert b == 4

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

# Logical Operators
assert not False
assert a == 2 and b == 4
assert a == 2 or c == 100

