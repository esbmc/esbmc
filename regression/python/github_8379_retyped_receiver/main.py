x = 3
x = "ab"
assert x.upper() == "AB"

y = 1
if (y := "cd"):
    assert y.upper() == "CD"
