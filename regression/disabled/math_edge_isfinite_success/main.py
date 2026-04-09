import math

assert math.isfinite(1.0)
assert not math.isfinite(math.inf)
assert not math.isfinite(math.nan)
