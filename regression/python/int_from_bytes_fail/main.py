bytes_data = b'\x00\x10'
x: int = int.from_bytes(bytes_data, 'big', False)

assert x == 15
