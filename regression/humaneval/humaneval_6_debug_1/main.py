# Debug 1: constant split only
if __name__ == "__main__":
    s: str = "(()()) ((()))"
    xs: list[str] = s.split(" ")
    assert xs[0] == "(()())"
    assert xs[1] == "((()))"
