# Tests FloorDiv (//) and Mod (%) with complex operands — all should TypeError
# Tests both complex//int and int//complex directions, same for %

# complex // int
try:
    r = complex(1, 2) // 2
    assert False
except TypeError:
    pass

# int // complex
try:
    r = 5 // complex(1, 0)
    assert False
except TypeError:
    pass

# complex % int
try:
    r = complex(1, 2) % 2
    assert False
except TypeError:
    pass

# int % complex
try:
    r = 5 % complex(1, 0)
    assert False
except TypeError:
    pass

# complex // complex
try:
    r = complex(1, 2) // complex(1, 0)
    assert False
except TypeError:
    pass

# complex % complex
try:
    r = complex(1, 2) % complex(1, 0)
    assert False
except TypeError:
    pass
