# min() with mixed int/float, float wins (Python promotes int to float)
c = [5, 2.5, 3]
assert min(c) == 2.5
