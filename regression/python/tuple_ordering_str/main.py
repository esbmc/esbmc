def before(s: str, n: int) -> bool:
    return (s, n) < ("m", 5)


assert (1, "b") < (3, "a")
assert ("a", 2) < ("b", 1)
assert ("ab", 1) > ("a", 9)
assert ("a", 2) <= ("a", 2)
assert not (("b", 1) < ("a", 2))
assert ("a", (1, "x")) < ("a", (1, "y"))

assert before("a", 9)
assert before("m", 4)
assert not before("m", 5)
assert not before("z", 0)
