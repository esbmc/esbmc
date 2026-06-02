flag = (0 == 1)
s = "9+0j" if flag else "0+9j"
z = complex(s)
assert z.real == 0.0
assert z.imag == 9.0
