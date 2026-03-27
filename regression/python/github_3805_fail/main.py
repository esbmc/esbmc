# 2 ** -10 = 0.0009765625, which is NOT >= 0.001 — assertion should fail
a = 2
b = -10
assert a**b >= 0.001
