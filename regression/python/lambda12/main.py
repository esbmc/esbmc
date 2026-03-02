def f(x: str) -> float:
    return {'+': lambda: 1.0}[x]()


f('+')
