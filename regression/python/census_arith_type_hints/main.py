from typing import Dict, List


def f(x: List[int], y: Dict[str, int]) -> int:
    return len(x) + len(y)


assert f([1], {}) == 1
