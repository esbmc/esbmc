# Tests (0+0j)**(-2) — should trigger ZeroDivisionError
# Zero base with negative exponent requires dividing 1/(0+0j)^2 = 1/(0+0j)

z = complex(0, 0) ** (-2)
