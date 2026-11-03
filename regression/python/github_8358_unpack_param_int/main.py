def total(ref: tuple[int, int]):
    x, y = ref
    return x + y


assert total((2**60 + 1, 2)) == 2**60 + 3
