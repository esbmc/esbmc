# Comprehensive pin of the whole V.1k (b) flipped binop surface under
# --python-irep2-adjust: integer + float arithmetic, all six comparisons,
# bitwise, shifts, and float division over member operands in one program.
class N:
    def __init__(self, a: int, b: int, p: float, q: float):
        self.a = a
        self.b = b
        self.p = p
        self.q = q


n = N(12, 5, 7.0, 2.0)

# integer arithmetic
assert n.a + n.b == 17
assert n.a - n.b == 7
assert n.a * n.b == 60

# integer comparisons (all six)
assert n.a > n.b
assert n.b < n.a
assert n.a >= 12
assert n.b <= 5
assert n.a == 12
assert n.a != n.b

# bitwise + shifts
assert (n.a & n.b) == 4
assert (n.a | n.b) == 13
assert (n.a ^ n.b) == 9
assert (n.b << 1) == 10
assert (n.a >> 2) == 3

# float arithmetic + division
assert n.p + n.q == 9.0
assert n.p - n.q == 5.0
assert n.p * n.q == 14.0
assert n.p / n.q == 4.0
