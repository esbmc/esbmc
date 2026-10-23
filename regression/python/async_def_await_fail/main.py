import asyncio


async def ratio(a: int, b: int) -> int:
    return a // b


async def main() -> None:
    r = await ratio(nondet_int(), nondet_int())


asyncio.run(main())
