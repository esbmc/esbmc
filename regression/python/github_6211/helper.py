def unused() -> int:
    return 1


y: int = unused()
assert y == 1
