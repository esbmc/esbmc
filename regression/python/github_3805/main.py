# Reproducer from issue #3805: a ** b with non-constant (local variable) exponent
# 2 ** -10 = 0.0009765625, which is < 0.001
a = 2
b = -10
assert a ** b < 0.001
