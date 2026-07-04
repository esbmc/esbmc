# str.encode("ascii")/decode("ascii") now fold for ASCII text: the byte
# representation is identical to UTF-8 (which was already accepted), so the
# ASCII-family encodings ("ascii", "us-ascii") are recognised too.
assert "hi".encode("ascii") == b"hi"
assert "hello".encode("us-ascii") == b"hello"
assert len("hello".encode("ascii")) == 5
assert "hi".encode(encoding="ascii") == b"hi"
assert "hi".encode("ascii").decode("ascii") == "hi"
# utf-8 and no-arg forms are unchanged.
assert "hi".encode("utf-8") == b"hi"
assert "hi".encode() == b"hi"
