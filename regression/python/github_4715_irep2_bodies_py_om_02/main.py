# Pins the Python dict operational-model allocation path under --irep2-bodies.
# A typed dict stores key/value objects whose storage is alloca'd from a
# sizeof(struct) constant; dropping #c_sizeof_type on the body round-trip
# (PR #5330) degraded the allocation to a byte array, producing alignment /
# NULL-pointer dereference failures on subscript reads. The reads must stay
# sound and correctly typed under the flag.
d: dict[int, float] = {1: 1.0}
d[2] = 3.5
assert isinstance(d[2], float)
assert d[2] == 3.5
assert d[1] == 1.0
assert not isinstance(d[2], int)
