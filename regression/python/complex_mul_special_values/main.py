import math

# NaN propagation through complex multiplication.
z_nan = complex(float("nan"), 0.0)
z_one = complex(1.0, 2.0)
w_nan = z_nan * z_one
assert math.isnan(w_nan.real)
assert math.isnan(w_nan.imag)

# NaN in imaginary part propagates through mul.
z_nan_i = complex(1.0, float("nan"))
w_nan2 = z_nan_i * complex(1.0, 0.0)
assert w_nan2.real == 1.0 or math.isnan(w_nan2.real)
assert math.isnan(w_nan2.imag)

# inf * finite = inf (some components).
z_inf = complex(float("inf"), 0.0)
w_inf = z_inf * complex(2.0, 0.0)
assert math.isinf(w_inf.real)

# Mul of purely imaginary: (0+1j)*(0+1j) = -1+0j.
z_i = complex(0, 1)
w_i = z_i * z_i
assert abs(w_i.real - (-1.0)) < 1e-10
assert abs(w_i.imag) < 1e-10

# Mul producing zero: (1+1j)*(1-1j) = 2+0j (no imaginary).
z_a = complex(1, 1)
z_b = complex(1, -1)
w_ab = z_a * z_b
assert abs(w_ab.real - 2.0) < 1e-10
assert abs(w_ab.imag) < 1e-10
