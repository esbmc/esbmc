import asyncio


async def f() -> int:
    await asyncio.sleep(0)
    return 1


asyncio.run(f())
