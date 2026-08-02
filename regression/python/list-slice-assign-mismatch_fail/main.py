# Extended slice assignment with a length mismatch raises ValueError:
# attempt to assign sequence of size 2 to extended slice of size 3.
l = [1, 2, 3, 4, 5, 6]
l[::2] = [10, 30]
print("unreachable under CPython")
