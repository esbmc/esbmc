b1: bool = nondet_bool()
b2: bool = nondet_bool()
s1: str = "short" if b1 else "longer string"
s2: str = "" if b2 else "x"
assert (len(s1) == 5 or len(s1) == 13) and (len(s2) == 0 or len(s2) == 1)
