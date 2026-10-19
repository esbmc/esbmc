def before(s: str, n: int) -> bool:
    return (s, n) < ("m", 5)


assert before("a", 9)
assert before("m", 5)
