x: float = 3.5
# 3.5 has a fractional part, so is_integer() is False; this assertion fails.
assert x.is_integer()
