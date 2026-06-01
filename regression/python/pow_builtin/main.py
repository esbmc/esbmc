assert pow(2, 10) == 1024
assert pow(2, -1) == 0.5
assert pow(2.0, 3.0) == 8.0
# 3-argument modular exponentiation (exact BigInt; 7**30 overflows float)
assert pow(2, 10, 1000) == 24
assert pow(7, 30, 13) == 12
assert pow(-2, 3, 5) == 2
