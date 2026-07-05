# ord() of a runtime string (variable, method result, or index) returns the
# code point of its single character without crashing goto_check.
def first_code(s: str) -> int:
    return ord(s)


def lower_code(s: str) -> int:
    return ord(s.lower())


def index_code(s: str) -> int:
    return ord(s[0])


if __name__ == "__main__":
    assert first_code("a") == 97
    assert lower_code("Z") == 122
    assert index_code("bc") == 98
