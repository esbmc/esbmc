from typing import Any


async def collect(xs: Any) -> Any:
    return [x async for x in xs]
