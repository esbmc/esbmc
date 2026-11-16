try:
    from esbmc import nondet_int
except ImportError:

    def nondet_int() -> int:
        return -2


class A:
    pass


assert type(2.5) is [int, float][1]
assert type(A()) is [int, A][-1]
assert type(2.5) is not [int, A][1]
assert [int, float][1] is type(2.5)
assert type(True) == [int, bool][1]

i = nondet_int()
if -3 <= i < 3:
    assert (type(2.5) is [int, float, str][i]) == (i == 1 or i == -2)
