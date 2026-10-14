flag = True
s = "3+1j" if flag else "1+3j"
z = complex(s)
assert z.real == 3.0
assert z.imag == 1.0
