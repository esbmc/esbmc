# Negative variant of list_not_truthiness: `not [1, 2, 3]` is False (a non-empty
# list is truthy), so asserting it must make verification FAIL. Guards against
# the IREP2 emptiness-comparison lowering collapsing to a constant.
xs = [1, 2, 3]
assert not xs
