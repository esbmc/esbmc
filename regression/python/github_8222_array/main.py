import array

a = array.array("B", [0, 0, 0])
assert a[0] == 0

b = array.array("h", [1, -2, 3])
b[1] = 7
b.append(-32768)
assert len(b) == 4
assert b[1] == 7 and b[3] == -32768
assert b.tolist() == [1, 7, 3, -32768]

s = 0
for x in b:
    s += x
assert s == -32757

assert not array.array("i")

try:
    a[0] = 256
except OverflowError:
    a[0] = 255
assert a[0] == 255

try:
    array.array("z")
except ValueError:
    pass
