def my_split(s: str, sep: str) -> list[str]:
    result = []
    word = ""
    for char in s:
        word = word + char
        assert word == "a" or "ab"
    assert word == "ab"
    result.append(word)
    result == ["ab"]
    return result


s: str = "ab"
l = my_split(s, ",")
assert l == ["ab"]
