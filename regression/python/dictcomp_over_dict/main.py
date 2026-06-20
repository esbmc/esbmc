# Dict comprehension iterating a dict (not a list of tuples): the C++ dict-comp
# handler rejects a dict/struct iterable, so this is desugared to a population
# for-loop. Covers a tuple target (unpacking the tuple keys), a single-name
# target (iterating the keys), iteration via d.keys(), and a generator filter.
d = {('A', 'B'): 3, ('C', 'D'): 7}
m = {v: 0 for u, v in d}
assert m['B'] == 0 and m['D'] == 0

n = {'a': 1, 'b': 2}
p = {k: 0 for k in n}
assert p['a'] == 0 and p['b'] == 0

# Iteration via d.keys().
q = {k: 0 for k in n.keys()}
assert q['a'] == 0 and q['b'] == 0

# Generator if-filter is preserved through the desugaring.
r = {k: 0 for k in n if k == 'a'}
assert r['a'] == 0 and len(r) == 1
