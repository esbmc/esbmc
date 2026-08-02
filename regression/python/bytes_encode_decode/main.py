def main() -> None:
    # str.encode() returns the (UTF-8/ASCII) bytes; bytes.decode() the str.
    assert "abc".encode() == b"abc"
    assert b"abc".decode() == "abc"

    # Variable receivers, and the result is the right type (len/index work).
    s = "hello"
    e = s.encode()
    assert len(e) == 5 and e[0] == 104  # 'h'

    b = b"world"
    d = b.decode()
    assert len(d) == 5 and d[0] == "w"

    # Explicit utf-8 encoding argument is accepted.
    assert "hi".encode("utf-8") == b"hi"
    assert b"hi".decode("utf-8") == "hi"

    # The encode/decode round-trip is the identity.
    assert s.encode().decode() == s


main()
