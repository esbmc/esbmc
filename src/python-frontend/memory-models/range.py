def ESBMC_range_next_(curr: int, step: int) -> int:
  return curr + step

def ESBMC_range_has_next_(curr: int, end: int, step: int) -> bool:
  return curr + step <= end
