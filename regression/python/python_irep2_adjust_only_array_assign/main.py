# A char[1] string literal assigned to a variable carrying the #5571
# fixed-width tuple-string representation. Under --python-irep2-adjust-only the
# assignment reached the solver unconverted and symex synthesised an
# array-to-array typecast that convert_typecast has no arm for.
def f(pairs: list[tuple[str, str]]) -> str:
    s = ""
    for u, v in pairs:
        s = v
    return s


assert f([("A", "B")]) == "B"
