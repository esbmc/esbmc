# Issue #3825: list lexicographic comparison with strings
quarter = ['A', 'B', 'C']

# Prefix: shorter < longer with same prefix
assert quarter < ['A', 'B', 'C', 'D']
assert ['A', 'B', 'C', 'D'] > quarter

# LtE / GtE
assert quarter <= ['A', 'B', 'C', 'D']
assert ['A', 'B', 'C', 'D'] >= quarter
assert quarter <= ['A', 'B', 'C']
assert quarter >= ['A', 'B', 'C']
