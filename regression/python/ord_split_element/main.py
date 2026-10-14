# ord() applied to an element of a str.split() result.
#
# A split() element is a runtime char* with no compile-time value, so ord()
# must read its first byte at runtime. Two paths are exercised: ord() on a
# subscript directly, and ord() on a variable bound to a split() element.


def last_char_code(txt: str) -> int:
    return ord(txt.split(' ')[-1])


def first_char_code(txt: str) -> int:
    part = txt.split(' ')[0]
    return ord(part)


if __name__ == "__main__":
    assert last_char_code("a b c") == 99  # 'c'
    assert first_char_code("a b c") == 97  # 'a'
