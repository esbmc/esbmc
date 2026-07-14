# Python's `%` is floored: the result takes the sign of the divisor, unlike
# C's truncated remainder (which takes the sign of the dividend). Verify all
# sign combinations and the floored-division identity, for both constant and
# symbolic operands.
assert -7 % 3 == 2
assert 7 % -3 == -2
assert -7 % -3 == -1
assert 7 % 3 == 1
assert -6 % 3 == 0


def check(a: int, b: int) -> None:
    if b != 0:
        r = a % b
        # |result| < |divisor| and result shares the divisor's sign (or is 0)
        assert (r == 0) or ((r > 0) == (b > 0))
        # floored-division identity holds
        assert (a // b) * b + r == a


check(-7, 3)
check(7, -3)
check(-7, -3)
check(7, 3)
