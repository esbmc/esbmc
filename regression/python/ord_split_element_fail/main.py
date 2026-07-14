# Negative variant of ord_split_element: ord() of the last split() element of
# "a b c" is 99 ('c'), so asserting 65 ('A') must fail.


def last_char_code(txt: str) -> int:
    return ord(txt.split(' ')[-1])


if __name__ == "__main__":
    assert last_char_code("a b c") == 65
