flag = (1 < 2)
s = "2+3j" if flag else "4+5j"
z = complex(s)
assert z.real == 2.0
assert z.imag == 3.0
