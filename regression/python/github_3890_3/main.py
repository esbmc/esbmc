def process(x: int) -> int:
    if type(x) == int:
        return x * 2
    return 0

result = process(5)
assert(result == 10)
