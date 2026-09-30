# CPython raises ValueError for any other byteorder; it is refused (#7945).
a = bytes([1, 0])
x = int.from_bytes(a, "middle")
