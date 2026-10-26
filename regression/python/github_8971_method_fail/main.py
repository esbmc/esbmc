class Probe:
    def is_zero(self, v):
        return v == 0


use_int = nondet_bool()
if use_int:
    x = nondet_int()
else:
    x = "text"
p = Probe()
assert p.is_zero(x) != (use_int and x == 0)
