class uint64(int):
    pass


ENDIANNESS = 'little'
SHUFFLE_ROUND_COUNT = 3


def compute_shuffled_permutation(index_count: uint64, seed: bytes):
    indices = [uint64(i) for i in range(index_count)]
    for current_round in range(SHUFFLE_ROUND_COUNT):
        round_bytes = current_round.to_bytes(1, ENDIANNESS)
        pivot = int.from_bytes(hash(seed + round_bytes)[0:8], ENDIANNESS) % index_count
        for i in range(index_count):
            flip = uint64((pivot + index_count - indices[i]) % index_count)
            position = int(max(indices[i], flip))
            position_bucket = position // 256
            source = hash(seed + round_bytes + position_bucket.to_bytes(4, ENDIANNESS))
            byte = source[(position % 256) // 8]
            bit = (byte >> (position % 8)) % 2
            indices[i] = flip if bit else indices[i]
    return indices


def compute_shuffled_index(index: uint64, index_count: uint64, seed: bytes) -> uint64:
    assert index < index_count
    return compute_shuffled_permutation(index_count, seed)[index]


seed = nondet_bytes(32)
result = compute_shuffled_index(uint64(0), uint64(2), seed)
assert result >= 2
