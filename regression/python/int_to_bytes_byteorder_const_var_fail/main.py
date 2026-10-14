BYTEORDER = 'big'

result = int.to_bytes(7, 4, BYTEORDER)
assert len(result) == 5
