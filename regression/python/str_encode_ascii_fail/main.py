# "hi".encode("ascii") is b"hi", not b"HI".
assert "hi".encode("ascii") == b"HI"
