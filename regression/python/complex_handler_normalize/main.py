# Tests the handler's normalize/promote path for integers and booleans
# in complex arithmetic contexts.

# int + complex: int is promoted to double then to complex.
assert (5 + complex(1, 2)) == complex(6, 2)

# complex + int
assert (complex(1, 2) + 5) == complex(6, 2)

# bool + complex: True is promoted through int -> double -> complex.
assert (True + complex(1, 2)) == complex(2, 2)

# complex + bool (False)
assert (complex(3, 4) + False) == complex(3, 4)

# int * complex
assert (3 * complex(2, 1)) == complex(6, 3)

# complex * int
assert (complex(2, 1) * 3) == complex(6, 3)

# bool * complex
assert (True * complex(5, 3)) == complex(5, 3)

# complex * bool (False): real and imag should be zero.
w_false = complex(5, 3) * False
assert w_false.real == 0.0
assert w_false.imag == 0.0

# int - complex
assert (10 - complex(3, 4)) == complex(7, -4)

# complex / int
assert (complex(6, 4) / 2) == complex(3, 2)

# int / complex
assert (5 / complex(1, 0)) == complex(5, 0)

# Equality with int: (1+0j) == 1 should be True.
assert complex(1, 0) == 1
assert not (complex(1, 0) != 1)

# Inequality with bool: complex(1,0) == True.
assert complex(1, 0) == True
assert not (complex(1, 0) != True)

# Equality with string: always not equal.
assert complex(1, 0) != "1"
assert not (complex(1, 0) == "1")  # type: ignore
