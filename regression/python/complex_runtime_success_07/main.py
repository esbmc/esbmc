
s = "1+2j"
flag = False
s = "4+5j" if flag else "6+7j"
z = complex(s)
assert z.real == 6.0
assert z.imag == 7.0

