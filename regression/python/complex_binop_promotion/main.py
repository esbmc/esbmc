assert (3 + 4j) == complex(3, 4)
assert ((1 + 2j) + 3) == complex(4, 2)
assert (3 + (1 + 2j)) == complex(4, 2)
assert ((1 + 2j) * (3 + 4j)) == complex(-5, 10)
assert ((1 + 0j) == 1)
assert ((1 + 2j) != (1 + 3j))

# Edge: comparison against non-numeric must follow Python semantics.
assert not ((1 + 0j) == "1")
assert ((1 + 0j) != "1")

# Edge: mixed float + complex promotion
assert (2.5 + (1 + 2j)) == complex(3.5, 2.0)

# Edge: UnaryOp over complex constant should preserve sign in literal conversion.
assert (-1j) == complex(0, -1)
