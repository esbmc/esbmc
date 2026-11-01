import numpy as np

class Counter:

    def __init__(self) -> None:
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return 7


c = Counter()
a = np.full(3, c.next())
assert c.n == 1
assert a[0] == 7 and a[2] == 7

narrow = np.full(2, c.next() + 0.5, dtype=np.int16)
assert c.n == 2
assert narrow[1] == 7 and narrow.itemsize == 2

a_m1 = np.full(2, 5, dtype=str)
assert a_m1[0] == "5"
b_m1 = np.full(2, 123, dtype=str)
assert b_m1[1] == "1" and b_m1.itemsize == 4
c_m1 = np.full(2, 123, dtype="U2")
assert c_m1[0] == "12"
d_m1 = np.array([12, 3], dtype=str)
assert d_m1[0] == "12" and d_m1[1] == "3"

literal_fill_m1 = np.full(2, "ab", dtype=str)
assert literal_fill_m1[1] == "a"
assert np.full(2, "ab")[1] == "ab"

a_m2 = np.full(3, (1, 2, 3))
assert a_m2[2] == 3
b_m2 = np.full((2, 2), [1, 2])
assert b_m2[0][0] == 1 and b_m2[1][1] == 2
c_m2 = np.full((2, 2), [[1], [2]], dtype=np.int8)
assert c_m2[0][1] == 1 and c_m2[1][0] == 2 and c_m2.itemsize == 1
c_m2[1][0] += 127
c_m2[1][0] += 127
assert c_m2[1][0] == 0
d_m2 = np.full((2, 3), [1.5, 2.5, 3.5], dtype=np.float32)
assert d_m2[1][2] == 3.5 and d_m2.nbytes == 24
e_m2 = np.full(2, [1.5])
assert e_m2[1] == 1.5
