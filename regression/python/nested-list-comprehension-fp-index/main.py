# A 2-D list comprehension records a single (homogeneous) element-type entry.
# Reading a constant outer index >= 1 and using the inner float element in FP
# arithmetic must resolve the float type, not fall through to a non-FP sort
# (previously crashed the SMT backend: get_exponent_width / Z3 "rm and fp
# sorts", #5129 — a residual of #5103/#5111 for the index>=1 case).

C = [[0.0 for _ in range(2)] for _ in range(2)]
i = 0
while i < 2:
    j = 0
    while j < 2:
        C[i][j] = 2.0 + 3.0
        j = j + 1
    i = i + 1

# Constant outer index >= 1 + FP arithmetic on the inner element.
x = C[1][0]
assert x - 5.0 == 0.0

# Row-extraction form of the same access.
r = C[1]
assert r[1] - 5.0 == 0.0
