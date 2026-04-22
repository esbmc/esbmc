# Tests conjugate mathematical identities with attribute access

# conj(a * b) == conj(a) * conj(b) (conjugate distributes over multiplication)
a = complex(1.0, 2.0)
b = complex(3.0, -1.0)
conj_prod = (a * b).conjugate()
prod_conj = a.conjugate() * b.conjugate()
assert conj_prod.real == prod_conj.real
assert conj_prod.imag == prod_conj.imag

# conj(a / b) == conj(a) / conj(b)
c = complex(6.0, 3.0)
d = complex(2.0, 1.0)
conj_div = (c / d).conjugate()
div_conj = c.conjugate() / d.conjugate()
assert conj_div.real == div_conj.real
assert conj_div.imag == div_conj.imag

# conj(z) * z is real (imag part == 0)
z1 = complex(2.0, 3.0)
prod = z1.conjugate() * z1
assert prod.imag == 0.0
assert prod.real == 13.0  # 2^2 + 3^2

# |conj(z)| == |z|
z2 = complex(5.0, 12.0)
assert abs(z2.conjugate()) == abs(z2)
