# Nested tuple-key unpacking in a dict.items() for-loop:
# for (u, v), w in d.items() must unpack the tuple key into its components.
d = {('A', 'B'): 3, ('C', 'D'): 7}
total = 0
seen_first = False
for (u, v), w in d.items():
    total += w
    if u == 'A' and v == 'B':
        seen_first = True
assert total == 10
assert seen_first
