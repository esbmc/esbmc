class uint256(int):
    pass


class BLSFieldElement(uint256):
    pass


Bytes32 = bytes


BLS_MODULUS = 123


KZG_ENDIANNESS = 'big'


def bls_field_to_bytes(x: BLSFieldElement) -> Bytes32:
    return int.to_bytes(x % BLS_MODULUS, 32, KZG_ENDIANNESS)

