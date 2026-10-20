# The element append stored is an int, not a str, so this assertion fails
# (#8950).
xs = []
xs.append(0)
assert isinstance(xs[0], str)
