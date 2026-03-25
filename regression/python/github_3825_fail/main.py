# Issue #3825: this assertion must FAIL
quarter = ['A', 'B', 'C']
# ['A', 'B', 'C'] is NOT less than ['A', 'B'] (it's greater, since it's longer with same prefix)
assert quarter < ['A', 'B']
