flag = True
s = "1+2j" if flag else "3+4j"
z = complex(s)
assert z.real == 1.0
assert z.imag == 2.0
