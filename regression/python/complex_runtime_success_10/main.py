flag = True
base = "5+4j" if flag else "4+5j"
z = complex(base)
assert z.real == 5.0
assert z.imag == 4.0
