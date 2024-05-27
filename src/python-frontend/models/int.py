class int:
  @classmethod
  def from_bytes(cls, bytes_data:bytes, big_endian:bool, signed:bool) -> int:
    result:int = 0
    index:int = 0
    step:int = 1
    byte:int = 0

    ## If little endian
    if big_endian == False:
        index:int = len(bytes_data) - 1
        step:int = -1

    bytes_len:int = len(bytes_data)

    while index >= 0 and index < bytes_len:
        byte:int = bytes_data[index]
        result:int = (result << 8) + byte
        index:int = index + step

    if signed and bytes_data[-1] & 128 == 128: # Check MSB of last byte
        is_negative:bool = True

    if signed and is_negative:
       result:int = result - (1 << (8 * len(bytes_data)))

    return result
