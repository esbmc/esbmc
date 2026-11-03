# The arguments of a coroutine call are evaluated even when it is not awaited.
async def ratio(a: int, b: int) -> int:
    return a // b


def main() -> None:
    b: int = nondet_int()
    ratio(10 // b, b)


main()
