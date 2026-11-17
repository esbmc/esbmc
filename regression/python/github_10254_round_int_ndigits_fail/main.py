# round(5, 1) is the int 5, not the float 5.0.
def rn(x: int) -> str:
    return str(round(x, 1))


assert rn(5) == "5.0"
