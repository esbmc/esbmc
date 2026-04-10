# Budget limit: negative exponent beyond budget uses exp/log fallback.
# z**(-17) = 1/(z**17) via exp(-17 * log(z)).
z = complex(0, 1)
w = z ** (-17)
# i^(-17) = 1/i^17 = 1/i = -i (since i^17 = i)
assert abs(w.real - 0.0) < 1e-4
assert abs(w.imag - (-1.0)) < 1e-4
