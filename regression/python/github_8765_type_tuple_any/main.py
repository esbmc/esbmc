TYPES = (str, int)

assert any(isinstance(1, a) for a in TYPES)
assert not any(isinstance(1.5, a) for a in TYPES)
assert all(isinstance(True, a) for a in (int, bool))
