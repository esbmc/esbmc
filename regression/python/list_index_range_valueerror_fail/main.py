# An uncaught ValueError from a missing element in the window fails verification.
l = [1, 2, 3]
l.index(9, 0, 2)
