# Test: Slicing com step
text = "0123456789"
slice1 = text[::2]  # "02468"
assert slice1 == "02468"
slice2 = text[1::2]  # "13579"
assert slice2 == "13579"
slice3 = text[::-1]  # reverso
assert slice3 == "9876543210"
