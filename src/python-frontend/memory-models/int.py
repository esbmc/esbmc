class int:
  def from_bytes(bytes_data:bytes, big_endian:bool, signed:bool) -> int:
    result = 0
    index = 0
    step = 1
    byte = 0

    ## If little endian
    if big_endian == False:
        index = len(bytes_data) - 1
        step = -1

    while index >= 0 and index < len(bytes_data):
       byte:int = bytes_data[index]
       result = (result << 8) + byte
       index = index + step

    if signed and bytes_data[-1] & 128 == 128: # Check MSB of last byte
       is_negative = True

    if signed and is_negative:
      result = result - (1 << (8 * len(bytes_data)))

    return result

bytes_data = b'\x00\x10'
result = int.from_bytes(bytes_data, True, False)
# assert result == 16

# bytes_data_2 = b'\xFF\xFF'
# result_2 = int.from_bytes(bytes_data_2, True, signed=True)
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


# print("All tests passed successfully!")