# Verifies hash_to_bls_field with hash() as nondet bytes, since real
# SHA-256 is intractable for BMC.

class uint256(int):
    pass


class BLSFieldElement(uint256):
    pass


KZG_ENDIANNESS = 'big'


def hash_to_bls_field(data: bytes) -> BLSFieldElement:
    """
    Hash ``data`` and convert the output to a BLS scalar field element.
    The output is not uniform over the BLS field.
    """
    hashed_data = hash(data)
    return BLSFieldElement(int.from_bytes(hashed_data, KZG_ENDIANNESS)) #% BLS_MODULUS)

