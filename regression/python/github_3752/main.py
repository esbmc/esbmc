a = -5580054495238291456
b = -7025463686093340672

z1 = complex(float(a) + float(b), 0)
z2 = complex(a + b, 0)

assert z1 == z2
