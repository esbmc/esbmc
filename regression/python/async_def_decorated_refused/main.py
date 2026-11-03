# A decorator receives the coroutine function, so the body is not run as a call.
def deco(fn):
    return fn


@deco
async def ratio(a: int, b: int) -> int:
    return a // b


def main() -> None:
    ratio(1, 0)


main()
