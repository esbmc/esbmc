# Pins the false proof: the dropped start argument folded index() to 0, so this
# claim verified. CPython's [1, 2, 1].index(1, 1) is 2.
assert [1, 2, 1].index(1, 1) == 0
