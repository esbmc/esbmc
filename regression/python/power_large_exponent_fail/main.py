# Test exponentiation with exponent > 64
# Const-eval should decline (return nullopt) and fallback to runtime evaluation

# Small exponent (within const-eval bounds)
a = 2 ** 10
assert a == 1023

# Exponent > 64 with base 1: const-eval declines, runtime handles it
b = 1 ** 100
assert b == 1

# Exponent > 64 with base 0: const-eval declines, runtime handles it
c = 0 ** 100
assert c == 0
