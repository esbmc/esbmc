bytes_data = b'\x00\x10'
x:int = int.from_bytes(bytes_data, 'big', False)
assert x == 16

# bytes_data_2 = b'\x00\x10'
# result_2:int = int.from_bytes(bytes_data_2, True, True)
# assert result_2 == -1

# bytes_data_3 = b'\x80'
# result_3 = int.from_bytes(bytes_data_3, True, signed=True)
# assert result_3 == -128

# bytes_data_4 = b'\xFF'
# result_4 = int.from_bytes(bytes_data_4, True, signed=False)
# assert result_4 == 255

# bytes_data_5 = b'\x00'
# result_5 = int.from_bytes(bytes_data_5, True, signed=False)
# assert result_5 == 0

# #Little-endian cases
# bytes_data_6 = b'\x10\x00'  # 16 in little-endian
# result_6 = int.from_bytes(bytes_data_6, False, signed=False)
# assert result_6 == 16

# bytes_data_7 = b'\xFF\xFF'  # -1 in little-endian
# result_7 = int.from_bytes(bytes_data_7, False, signed=True)
# assert result_7 == -1

# bytes_data_8 = b'\x80'  # -128 in little-endian
# result_8 = int.from_bytes(bytes_data_8, False, signed=True)
# assert result_8 == -128

# bytes_data_9 = b'\xFF'  # 255 in little-endian
# result_9 = int.from_bytes(bytes_data_9, False, signed=False)
# assert result_9 == 255

# bytes_data_10 = b'\x00'  # 0 in little-endian
# result_10 = int.from_bytes(bytes_data_10, False, signed=False)
# assert result_10 == 0
