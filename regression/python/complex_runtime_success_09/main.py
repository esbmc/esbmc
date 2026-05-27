flag = False
s = "8+8j" if flag else "6+2j"
z = complex(s)
assert z.real == 6.0
assert z.imag == 2.0
