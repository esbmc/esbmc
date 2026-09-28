# A parameter that shadows the module constant is not the constant (#7542).
ENDIANNESS = 'little'


def decode(data: bytes, ENDIANNESS: str) -> int:
    return int.from_bytes(data, ENDIANNESS)


assert decode(bytes([1, 0]), 'big') == 256
