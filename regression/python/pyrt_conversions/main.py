# A builtin type used as a call converts, rather than building an instance of
# the type. int() truncates toward zero, as CPython does.
assert int(2.7) == 2
assert int(-2.7) == -2
assert int(5) == 5
assert int(True) == 1
assert float(3) == 3.0
assert float(2.5) == 2.5
assert str("ab") == "ab"
assert len(list()) == 0
assert len(dict()) == 0
assert len(tuple()) == 0
