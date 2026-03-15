# Issue #3825: more string ordering cases
# First differing element determines outcome
assert ['A', 'C'] > ['A', 'B']
assert ['A', 'B'] < ['A', 'C']
assert ['Z'] > ['A', 'B', 'C']
assert ['A'] < ['Z']

# Longer list is greater when prefix matches
assert ['January', 'February'] < ['January', 'February', 'March']
assert ['January', 'February', 'March'] > ['January', 'February']
