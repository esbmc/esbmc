def strip_spaces_and_join(s: str) -> str:
    chars = []
    for c in s:
        if c != ' ':
            chars.append(c)
    return ''.join(chars)

if __name__ == "__main__":
    assert strip_spaces_and_join("( )") == "()"
    assert strip_spaces_and_join("( ( ) )") == "(())"
    print("OK")
