# The byteorder string reaches the model unchanged, however the call is
# spelled (#7945).
f = int.from_bytes
a = bytes([1, 0])
assert f(a, "little") == 256
