# round(int, n) is an int, so it prints without a fractional part.
def rn(x: int) -> str:
    return str(round(x, 1))


assert rn(5) == "5"
