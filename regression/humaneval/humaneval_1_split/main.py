def split_paren_groups(s: str) -> list[str]:
    result = []
    current = []
    depth = 0
    for c in s:
        if c == '(':
            depth += 1
            current.append(c)
        elif c == ')':
            depth -= 1
            current.append(c)
            if depth == 0:
                result.append(''.join(current))
                current.clear()
    return result

if __name__ == "__main__":
    assert split_paren_groups("() (())") == ["()", "(())"]
    assert split_paren_groups("(()) ()") == ["(())", "()"]
    print("OK")
