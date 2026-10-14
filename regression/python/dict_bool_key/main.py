# A dict with boolean keys now resolves subscript, membership, and .get()
# correctly. Previously the bool key was stored widened to a long (8 bytes)
# but every lookup query used bool's native 1-byte size, so the OM size
# check never matched. Bool as a dict *value* was always fine.
d = {True: 1, False: 0}
assert d[True] == 1
assert d[False] == 0
assert True in d
assert False in d
assert d.get(True) == 1
assert d.get(False) == 0
