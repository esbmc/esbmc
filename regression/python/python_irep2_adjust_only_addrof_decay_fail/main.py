# Negative counterpart: same decayed string-literal address in the model's raise
# path, with a false assertion. The hop-off must still reach the solver and
# report the violation.
import math


def safe_sqrt(x: float) -> float:
    try:
        return math.sqrt(x)
    except ValueError:
        return -1.0


assert safe_sqrt(-1.0) == 99.0
