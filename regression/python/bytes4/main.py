data = b'\x00\x01\x02\x00'
assert len(data) == 4
assert data[0] == 0
assert data[1] == 1
assert data[2] == 2
assert data[3] == 0
