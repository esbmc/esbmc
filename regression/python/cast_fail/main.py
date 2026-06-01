a = 60
b = 5

sum = float(a + b)
assert sum == 65.0

# sum is rebound from float to str (chr). The straight-line retyping fix
# (#4770) preserves the real value chr(65) == 'A', so asserting 'B' must FAIL.
# Guards against the rebind being dropped or made nondet (which would not
# deterministically refute this).
sum = chr(int(sum))
assert sum == 'B'
