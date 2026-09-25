# Pins the false SUCCESSFUL: ESBMC folded the 6-significant-digit ostream form,
# so this false claim verified. CPython gives "1.23456789".
assert "{}".format(1.23456789) == "1.23457"
