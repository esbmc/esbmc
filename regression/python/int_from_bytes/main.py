bytes_data = b'\x00\x10'

x:int = int.from_bytes(bytes_data, 'big', False)
assert x == 16

y:int = int.from_bytes(bytes_data, 'little', False)
assert y == 4096

