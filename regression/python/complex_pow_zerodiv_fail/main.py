# (0+0j) ** -1 should cause ZeroDivisionError
# because it computes 1 / (0+0j) which is zero division.
z = complex(0, 0)
w = z ** (-1)
