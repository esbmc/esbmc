def clamp_len(xs: list[int], cap: int) -> int:
    return max(0, min(len(xs), cap))

assert clamp_len([1, 2, 3], 2) == 2
assert clamp_len([], 5) == 0
