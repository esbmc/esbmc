# A coroutine that is never awaited does not run its body.
async def ratio(a: int, b: int) -> int:
    return a // b


def main() -> None:
    a: int = nondet_int()
    b: int = nondet_int()
    ratio(a, b)


main()
