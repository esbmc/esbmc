
real_txt = "8"
imag_txt = "9"
flag = True
s = real_txt + "+" + imag_txt + "j" if flag else "0+0j"
z = complex(s)
assert z.real == 8.0
assert z.imag == 9.0

