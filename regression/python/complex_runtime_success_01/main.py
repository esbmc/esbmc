
flag = False
s = "1+2j" if flag else "3+4j"
z = complex(s)
assert z.real == 3.0
assert z.imag == 4.0

