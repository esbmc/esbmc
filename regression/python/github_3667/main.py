# Regression test for GitHub issue #3667:
# ESBMC crashed with "List variable not found" when calling a list method
# (e.g. append) on a subscript of a nested list after list.copy().

nested = [[1], [2]]
shallow = nested.copy()
nested[0].append(99)

# Shallow copy shares the inner lists, so both views see the mutation.
assert nested[0][1] == 99
assert shallow[0][1] == 99
