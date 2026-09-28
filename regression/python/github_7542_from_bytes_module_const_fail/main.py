# A byteorder named by a module constant, as the consensus specs write it
# (#7542), folds like the literal it is bound to.
ENDIANNESS = 'little'
KZG_ENDIANNESS = 'big'

a = bytes([1, 0])
assert int.from_bytes(a, ENDIANNESS) == 256
assert int.from_bytes(a, byteorder=KZG_ENDIANNESS) == 256
