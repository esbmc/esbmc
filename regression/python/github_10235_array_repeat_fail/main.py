from array import array

a = array("d", [0]) * 3
a[1] = 1.5
assert a[2] == 1.5
