# Storing an int into a float slot keeps the slot's float type_id, so
# isinstance answers from the old type and this assertion is missed (#8950).
xs = [1, 2.5]
xs[1] = 3
assert not isinstance(xs[1], int)
