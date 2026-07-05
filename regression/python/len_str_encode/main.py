# len(s.encode()) on an inline call now sizes the bytes by element count.
# The encoded bytes are a wide-int array whose zero high bytes made strlen
# stop early, so the inline form (unlike a materialised variable) returned a
# wrong length.
assert len("abc".encode()) == 3
assert len("".encode()) == 0
assert len("hello world".encode()) == 11
b = "abcde".encode()
assert len(b) == 5
