def ESBMC_range_next_(curr: int, step: int) -> int:
    return curr + step


def ESBMC_range_has_next_(curr: int, end: int, step: int) -> bool:
    if step > 0:
        return curr < end
    elif step < 0:
        return curr > end
    else:
        return False  # step == 0 is invalid
