# all() with all True values
assert all([True, True, True]) == True

# all() with a False value
assert all([True, False, True]) == False

# all() with empty list returns True (vacuous truth)
assert all([]) == True

# Truthy integers
assert all([1, 2, 3]) == True

# Falsy zero
assert all([1, 0, 3]) == False

# None is falsy
assert all([None, True]) == False
assert all([True, None]) == False

# Mixed falsy values
assert all([False, None, 0]) == False

# Single element
assert all([True]) == True
assert all([False]) == False
assert all([1]) == True
assert all([0]) == False

# Nested non-empty containers are truthy (non-NULL pointer)
assert all([[False]]) == True
assert all([[None]]) == True
assert all([[1, 2]]) == True

# Negative integers are truthy
assert all([-1]) == True
assert all([-1, -2, -3]) == True
assert all([-1, 0]) == False

# Large integers are truthy
assert all([1000000]) == True
assert all([1000000, 0]) == False

# Float truthiness: 0.0 is falsy, non-zero is truthy
assert all([1.0]) == True
assert all([0.0]) == False
assert all([1.0, 2.5]) == True
assert all([1.0, 0.0]) == False
assert all([-1.0]) == True
assert all([-0.1]) == True

# Mixed int and float
assert all([1, 2.0, 3]) == True
assert all([1, 0.0, 3]) == False

# True == 1 and False == 0 in Python
assert all([True, 1]) == True
assert all([False, 0]) == False
assert all([True, False]) == False
assert all([1, False]) == False

# None mixed with integers
assert all([1, None]) == False
assert all([None, 1]) == False

# Single None
assert all([None]) == False

# Multiple None values
assert all([None, None]) == False

# Non-empty string is truthy (ESBMC treats any string as truthy via else branch)
assert all(["a"]) == True
assert all(["hello"]) == True

# Non-empty list as element is truthy (non-NULL pointer)
assert all([[1]]) == True
assert all([[0]]) == True
assert all([[True]]) == True

# Non-empty tuple as element is truthy (struct, falls to else branch -> true)
assert all([(1, )]) == True
assert all([(0, )]) == True
assert all([(False, )]) == True

# Non-empty dict as element is truthy (struct, falls to else branch -> true)
assert all([{1: 2}]) == True

# float('inf') and float('-inf') are truthy
assert all([float('inf')]) == True
assert all([float('-inf')]) == True

# Combination of valid truthy elements
assert all([True, 1, -1, 1.5, float('inf')]) == True

# Short-circuit equivalents: first falsy element makes result False
assert all([0, True, True]) == False
assert all([True, 0, True]) == False
assert all([True, True, 0]) == False
assert all([None, 1, 1]) == False
assert all([1, None, 1]) == False
assert all([1, 1, None]) == False

# Empty containers as elements are falsy in Python
assert all([[]]) == False
assert all([[], []]) == False
assert all([1, []]) == False
assert all([()]) == False
assert all([""]) == False
assert all([{}]) == False
assert all([1, [], 2]) == False
