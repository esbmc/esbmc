# Negative variant of sorted_tuples_concrete: the list is sorted correctly, but
# the assertion claims the wrong order. Confirms the compile-time sort produces
# the precise lexicographic result (a wrong claim is refuted).

s = sorted([(3, 1), (1, 2)])
assert s[0] == (3, 1)
