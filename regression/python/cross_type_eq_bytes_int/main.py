b: bytes = b'A'
# Whole-bytes vs int: distinct categories ("bytes" vs "numeric") → cross-type fold
assert (b == 65) == False
assert (b != 65) == True
# Indexed bytes vs int: in CPython `b[0]` is an int, so this must hold.
# Locks in that bytes elements are modeled as numerics (long_long_int), not as
# 8-bit chars — otherwise the 8-bit-int-as-string heuristic in
# get_python_type_category would wrongly fold this comparison to False.
assert b[0] == 65
assert (b[0] != 65) == False
