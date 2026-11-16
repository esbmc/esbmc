class HelperError(Exception):
    pass

def go(x: int) -> int:
    if x > 0:
        raise HelperError("h")
    return x
