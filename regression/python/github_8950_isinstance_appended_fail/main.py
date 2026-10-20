# The int that append stored is an int, so this assertion fails (#8950).
xs = []
xs.append(0)
assert not isinstance(xs[0], int)
