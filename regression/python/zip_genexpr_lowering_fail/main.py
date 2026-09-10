result = list(a + b for a, b in zip([1, 2, 3], [10, 20, 30]))
assert result == [11, 22, 31]
