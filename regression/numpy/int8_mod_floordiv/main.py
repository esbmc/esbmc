import numpy as np

a = np.add(7, 0, dtype=np.int8)
b = np.add(3, 0, dtype=np.int8)
assert a % b == 1
assert a // b == 2

c = np.add(-7, 0, dtype=np.int8)
assert c // b == -3
assert c % b == 2

x = np.array([100, -100], dtype=np.int8)
u = np.array([200, 100], dtype=np.uint8)
s = np.array([1000, -1000], dtype=np.int16)
f = np.array([1.5, 2.5], dtype=np.float32)
h = np.array([1.5, 2.5], dtype=np.float16)

wrapped = x + x
assert wrapped[0] == -56 and wrapped.itemsize == 1
mixed = x + u
assert mixed[0] == 300 and mixed.itemsize == 2
weak = x + 1
assert weak[0] == 101 and weak.itemsize == 1
widened = x * 2.5
assert widened[0] == 250.0 and widened.itemsize == 8
halves = x / 2
assert halves[1] == -50.0 and halves.itemsize == 8
single = f / 2
assert single[0] == 0.75 and single.itemsize == 4
half = h * h
assert half[1] == 6.25 and half.itemsize == 2
short_float = s + f
assert short_float[0] == 1001.5 and short_float.itemsize == 4
byte_half = h + x
assert byte_half[0] == 101.5 and byte_half.itemsize == 2
remainder = x % 3
floored = x // 3
power = x**2
assert remainder[1] == 2 and floored[1] == -34 and power[0] == 16
masked = x & u
shifted = x << 1
assert masked[0] == 64 and masked.itemsize == 2 and shifted[0] == -56

assert x[0] + x[0] == -56
assert u[0] + x[0] == 300
assert float(f[0] + 0.1) != 1.6
assert x[0] * 2.5 == 250.0
