flag = True
s = "0+1j" if flag else "1+0j"
z = complex(s)
assert z.real == 0.0
assert z.imag == 1.0
