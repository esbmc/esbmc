from fwd import g, total

assert g() == "s"
assert g(1) == 1
assert total(1) == 1
assert total(1, 2, 3) == 6
