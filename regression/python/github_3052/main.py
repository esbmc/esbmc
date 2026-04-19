def foo(s: str, by: bytes, b: bool) -> None:
    if b:
        r = s.encode('utf-8')
    else:
        r = by
