# Dict comprehension iterating a dict (not a list of tuples) with a tuple
# target that unpacks the tuple keys: the C++ dict-comp handler rejects a
# dict/struct iterable, so this is desugared to a population for-loop.  The
# single-name / d.keys() / if-filter variants live in dictcomp_over_dict_name
# and dictcomp_over_dict_keys (kept separate so each test stays well under the
# 120s cap: combining the comprehensions makes incremental-bmc blow up).
d = {('A', 'B'): 3, ('C', 'D'): 7}
m = {v: 0 for u, v in d}
assert m['B'] == 0 and m['D'] == 0
