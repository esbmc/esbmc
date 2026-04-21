# nondet_complex() should be able to trigger verification failure.
# A nondet complex is not guaranteed to equal any specific value.

z = nondet_complex()
assert z == complex(0, 0)
