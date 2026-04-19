def not_called(s: str) -> None:
    b: bytes = s.encode("utf-8")  # Unsupported but not called
    
assert True  # Should verify successfully
