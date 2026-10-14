# ord() of a runtime string returns the code point of its first character.
def first_code(s: str) -> int:
    return ord(s)


if __name__ == "__main__":
    assert first_code("a") == 98  # wrong: ord("a") == 97
