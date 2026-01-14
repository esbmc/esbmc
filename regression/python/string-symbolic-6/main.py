# Test: Slicing de string - deve PASSAR
text = "Python"
slice1 = text[1:4]  # "yth"
assert slice1 == "yth"
assert len(slice1) == 3
slice2 = text[:2]  # "Py"
assert slice2 == "Py"
