def called(s: str) -> None:
    b: bytes = s.encode("utf-8")  # Unsupported and IS called


called("test")
