def fail() -> int:
    raise ValueError("Error")
    return 0


result = fail()
