# Verifies compute_shuffled_index and compute_shuffled_permutation, the
# phase0 spec's swap-or-not shuffle. hash() is nondet bytes and
# SHUFFLE_ROUND_COUNT/index_count are reduced for tractability; the
# per-bucket hash cache in the spec's compute_shuffled_permutation is
# dropped since it is a pure performance optimisation over recomputing
# the same hash.


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


index_count = nondet_int()
__ESBMC_assume(index_count >= 1 and index_count <= 2)
index = nondet_int()
__ESBMC_assume(index >= 0 and index < index_count)
seed = nondet_bytes(32)

result = compute_shuffled_index(uint64(index), uint64(index_count), seed)
assert result >= 0
assert result < index_count
