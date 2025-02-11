lst = [1, 2, 3, "four", 5]
assert all(isinstance(x, int) for x in lst), "List contains non-integer elements!"
