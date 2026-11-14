# The converter folds this call on a constant string into a list literal
# instead of calling the function, so the literal must be a typed list too.
def parse_nested_parens(s: str) -> list[int]:
    result: list[int] = []
    depth = 0
    best = 0
    for c in s:
        if c == "(":
            depth += 1
            if depth > best:
                best = depth
        elif c == ")":
            depth -= 1
        else:
            result.append(best)
            depth = 0
            best = 0
    result.append(best)
    return result


r: list[int] = parse_nested_parens("(()) ()")
assert r[0] == 2
