# Issue #10579: a float rendered by an f-string, joined with literals.
def g(b: float) -> str:
    return f"<{b}>"


assert g(1.0) == "<1.00>"
