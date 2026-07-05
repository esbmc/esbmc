# Dict comprehension iterating a dict via d.keys(), and with a generator
# if-filter preserved through the desugaring to a population for-loop.  The
# bare-dict tuple/name variants live in dictcomp_over_dict (kept separate so
# each test stays well under the 120s cap).
n = {'a': 1, 'b': 2}

# Iteration via d.keys().
q = {k: 0 for k in n.keys()}
assert q['a'] == 0 and q['b'] == 0

# Generator if-filter is preserved through the desugaring.
r = {k: 0 for k in n if k == 'a'}
assert r['a'] == 0 and len(r) == 1
