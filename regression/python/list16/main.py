quarter = ['January', 'February', 'March']

# Membership
assert 'January' in quarter
assert 'April' not in quarter

# Equality and inequality
assert quarter == ['January', 'February', 'March']
assert quarter != ['March', 'February', 'January']

# Ordering (lexicographic)
assert quarter < ['January', 'February', 'March', 'April']
assert quarter > ['January', 'February']
assert ['April'] < quarter
assert ['Zebra'] > quarter

# Edge case: empty list comparisons
empty = []
assert empty < quarter
assert quarter > empty

