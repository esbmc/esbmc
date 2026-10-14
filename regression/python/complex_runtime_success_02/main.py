flag = True
s = "5+6j" if flag else "7+8j"
z = complex(s)
assert z.real == 5.0
assert z.imag == 6.0
