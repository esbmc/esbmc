# Edge 1: string comes from conditional expression (not a direct literal call arg).
flag = True
s = "1+2j" if flag else "3+4j"
z = complex(s)
assert z.real == 1.0
assert z.imag == 2.0

# Edge 2: runtime variable containing malformed string still raises ValueError.
bad = "++1j"
raised_value = False
try:
    complex(bad)
except ValueError:
    raised_value = True
assert raised_value
