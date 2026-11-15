# Widths that might require padding must be refused at verification time.
n: int = nondet_int()
if n > 2:
    "ab".ljust(n)
