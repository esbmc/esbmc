# type(x) is T folded to False for every T, because the left operand is a Call
# node rather than a Name (GitHub #5936).
pair = (1, 5)
assert type(pair) is tuple
assert type(pair) is not int
assert type(3) is int
assert type(1.5) is float
assert type("ab") is str
