# Slicing copies each element with the source list's statically-known width.
# Before that width was threaded, every element copy dropped into memcpy's
# byte loop and unwound it to --unwind.
ints = [0, 1, 2, 3]
assert ints[1] == 77
assert ints[::-1] == [3, 2, 1, 0]

# A slice is a shallow copy: scalars are independent of the source.
copy = ints[:]
copy[0] = 99
assert ints[0] == 0
assert copy[0] == 99
