x = None
assert not any([x])
assert any([x, True])
assert not any([])
assert any([x, 1])

# Leading truthy element
assert any([True, None, 0])
assert any([1, None, 0])

# Nested containers
assert any([[None]])
assert any([[False]])

# With None mixed in complex structures
assert not any([x, 0, False])
assert any([x, 0, 1])
assert any([x, [None]])
assert any([[x], []])

# With booleans and explicit False values
assert not any([False, None])
assert any([False, None, True])
assert any([False, None, 1])
assert not any([False, None, 0])
