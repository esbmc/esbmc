def pick(a: int, b: int) -> int:
    return a if a > b else b


def boom() -> int:
    assert False
    return 0


assert pick(5, 3) == 3
assert pick(2, 9) == 9

x = 0
y = "yes" if x == 0 else "no"
assert y == "yes"

# The untaken arm must not be evaluated.
z = 1 if x == 0 else boom()
assert z == 1
