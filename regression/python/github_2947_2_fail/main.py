# Basic tuple
t1: tuple = (1, 2, 3)
assert not isinstance(t1, tuple)
assert t1 == (2, 2, 3)
assert len(t1) == 4
assert t1[0] == 2 and t1[1] == 2 and t1[2] == 3

# Mixed types
t2: tuple = (42, "hello", 3.14, True)
assert isinstance(t2, tuple)
assert len(t2) == 4
assert t2[0] == 42
assert t2[1] == "hello"
assert abs(t2[2] - 3.14) < 1e-9
assert t2[3] is True

# Empty tuple
t3: tuple = ()
assert isinstance(t3, tuple)
assert len(t3) == 0
assert t3 == ()

# Single element
t4: tuple = (5, )
assert isinstance(t4, tuple)
assert len(t4) == 1
assert t4[0] == 5

# Nested
t5: tuple = ((1, 2), (3, 4))
assert isinstance(t5, tuple)
assert len(t5) == 2
assert isinstance(t5[0], tuple) and isinstance(t5[1], tuple)
assert t5[0] == (1, 2)
assert t5[1] == (3, 4)
assert t5 == ((1, 2), (3, 4))
