# Negative counterpart: the same container-stored lambda call with a false
# assertion. The dereferenced callee must reach the solver and report it.
def pick(key: str) -> float:
    return {'+': lambda: 1.0, '-': lambda: 2.0}[key]()


assert pick('+') == 99.0
