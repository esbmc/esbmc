def offset() -> int:
    return 1


for i, (a, b) in enumerate([(1, 2)], offset()):
    assert a + b == 3
