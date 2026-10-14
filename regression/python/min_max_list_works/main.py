# min()/max() over a list or tuple (including lists of strings) works; this
# pins the boundary of the string-argument limitation in min_max_string_knownbug.
assert min([3, 1, 2]) == 1
assert max([3, 1, 2]) == 3
assert min(["b", "a", "c"]) == "a"
assert max(["b", "a", "c"]) == "c"
assert min((5, 2, 8)) == 2
assert max((5, 2, 8)) == 8
