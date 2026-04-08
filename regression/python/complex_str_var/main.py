# str() on complex variable (not inline constructor).

z = complex(2, -3)
s = str(z)
assert s == "(2-3j)"
