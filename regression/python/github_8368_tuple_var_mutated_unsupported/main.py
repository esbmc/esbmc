# The list held in p is changed through p before the unpacking, so the
# element types recorded for its literal are not trusted and it is refused.
p = ([1, 2], 3)
p[0][0] = "s"
(x, y), m = p
assert x == "s"
