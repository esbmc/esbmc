# Negative variant of list_sorted_dict (GitHub #4790).
#
# list(d) yields the keys in insertion order, so list({7:70, 9:90})[1] is 9,
# not 7. The subscript now reads the correct key value, so asserting it equals
# 7 must be refuted (before the fix the subscript read a wrong constant, which
# would refute this for the wrong reason).
d = {7: 70, 9: 90}
ks = list(d)
assert ks[1] == 7
