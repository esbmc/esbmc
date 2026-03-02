quarter = ['January', 'February', 'March']

# Basic index checks
assert quarter[0] == 'January'
assert quarter[1] == 'February'
assert quarter[2] == 'March'
assert quarter[-1] == 'March'  # negative indexing -  (last element)
assert quarter[-2] == 'February'  # (second to last)

# Length
assert len(quarter) == 3
