def check(b: int) -> int:
    x: int = 0
    if b != 0 and (10 // b) > 0:
        x = 1
    return x


z: int = check(0)
