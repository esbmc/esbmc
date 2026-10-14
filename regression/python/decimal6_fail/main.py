from decimal import Decimal

# copy_abs should clear sign, so asserting sign==1 should fail
d: Decimal = Decimal("-3.14")
r: Decimal = d.copy_abs()
assert r._sign == 1
