def sink(v: int) -> int:
    return v


def run(n: int, d: int) -> None:
    sink((n + d // n) // 2)


run(8, 4)
