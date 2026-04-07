# Tests chained attribute access: conjugate().real, conjugate().imag
# Uses intermediate variables since ESBMC doesn't resolve chained method+attr inline

z1 = complex(3.0, 4.0)

# conjugate().real should be same as original .real
c1 = z1.conjugate()
cr = c1.real
assert cr == 3.0

# conjugate().imag should be negated
ci = c1.imag
assert ci == -4.0

# Double conjugate via chained access
z2 = complex(-1.0, 5.0)
c2 = z2.conjugate()
dc = c2.conjugate()
assert dc.real == -1.0
assert dc.imag == 5.0

# Chained on purely imaginary
z3 = complex(0.0, 7.0)
c3 = z3.conjugate()
assert c3.real == 0.0
assert c3.imag == -7.0

# Chained on purely real (conjugate is identity)
z4 = complex(9.0, 0.0)
c4 = z4.conjugate()
assert c4.real == 9.0
assert c4.imag == 0.0
