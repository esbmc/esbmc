# Dict comprehension iterating a dict with a single-name target (iterating the
# keys): the C++ dict-comp handler rejects the dict/struct iterable, so this is
# desugared to a population for-loop.  The tuple-target and d.keys()/if-filter
# variants live in dictcomp_over_dict and dictcomp_over_dict_keys (kept separate
# so each test stays well under the 120s cap).
n = {'a': 1, 'b': 2}
p = {k: 0 for k in n}
assert p['a'] == 0 and p['b'] == 0
