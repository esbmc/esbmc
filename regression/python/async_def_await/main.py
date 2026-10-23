import asyncio


async def ratio(a: int, b: int) -> int:
    return a // b


async def main() -> None:
    b: int = nondet_int()
    __ESBMC_assume(b > 0 and b < 10)
    r = await ratio(10 * b, b=b)
    assert r == 10


asyncio.run(main())
