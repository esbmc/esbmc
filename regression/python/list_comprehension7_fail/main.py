a = b = [i for i in range(3)]
assert a == b
assert len(a) == 3
assert len(b) == 2  # This assertion will fail because len(b) is actually 3, not 2.
