x: str = "abc"
y: str = x[::-1]
assert y == "cba"

# Reverse of a palindrome is itself
p: str = "aba"
q: str = p[::-1]
assert q == "aba"
assert p == q

# Single char reversal
s: str = "x"
assert s[::-1] == "x"

# Combined: slice then reverse
t: str = "abcde"
u: str = t[1:]
v: str = u[::-1]
assert v == "edcb"
